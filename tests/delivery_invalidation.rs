//! Delivery invalidation across final release, distinct from retained graph identity.

extern crate alloc;

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
#[cfg(feature = "imperative-signal-evaluation")]
use jonmo::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, SameSignalIdentity, ScalarSignalMaterializer,
    ScalarSlotCommand,
    graph::{downcast_signal_value, run_signal_subgraph_once},
};
use jonmo::{OutputDelivery, ReplayInitialization, VecSignalMaterializer, prelude::*};

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct Producer;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct Consumer;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct Delivery;

#[derive(Resource, Default)]
struct Pending(Option<i32>);

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(
        JonmoPlugin::new::<Producer>()
            .with_schedule::<Consumer>()
            .with_schedule::<Delivery>(),
    );
    app.init_resource::<Pending>();
    app.update();
    app
}

fn pending(In(()): In<()>, mut pending: ResMut<Pending>) -> Option<i32> {
    pending.0.take()
}

fn queue_pushes(
    app: &mut App,
    values: &MutableVec<i32>,
    source: &SignalClaim,
    drain: Drain,
    inputs: impl IntoIterator<Item = i32>,
) {
    let _ = source;
    for value in inputs {
        values.write(app.world_mut()).push(value);
        match drain {
            Drain::Scheduled => app.world_mut().run_schedule(Producer),
            #[cfg(feature = "imperative-signal-evaluation")]
            Drain::Polled => {
                let batch = run_signal_subgraph_once(app.world_mut(), source.entity())
                    .and_then(downcast_signal_value::<Vec<VecDiff<i32>>>)
                    .expect("one pending push");
                assert!(matches!(batch.as_slice(), [VecDiff::Push { value: actual }] if *actual == value));
            }
        }
    }
}

fn queue(app: &mut App, source: &SignalClaim, drain: Drain, values: impl IntoIterator<Item = i32>) {
    let _ = source;
    for value in values {
        app.world_mut().resource_mut::<Pending>().0 = Some(value);
        match drain {
            Drain::Scheduled => app.world_mut().run_schedule(Producer),
            #[cfg(feature = "imperative-signal-evaluation")]
            Drain::Polled => {
                assert_eq!(
                    run_signal_subgraph_once(app.world_mut(), source.entity()).and_then(downcast_signal_value::<i32>),
                    Some(value),
                    "each producer evaluation queues one separate input"
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Drain {
    Scheduled,
    #[cfg(feature = "imperative-signal-evaluation")]
    Polled,
}

#[derive(Resource)]
struct RetainedTerminal(Broadcast<()>);

#[derive(Resource, Default)]
struct ReentrantState {
    claim: Option<SignalClaim>,
    armed: bool,
    reregister: bool,
    released: bool,
    registered: bool,
    calls: Vec<String>,
}

fn reentrant_callback(In(batch): In<Vec<VecDiff<i32>>>, world: &mut World) {
    world.resource_mut::<ReentrantState>().calls.push(format!("{batch:?}"));
    if matches!(batch.as_slice(), [VecDiff::Push { value: 1 }])
        && core::mem::take(&mut world.resource_mut::<ReentrantState>().armed)
    {
        let claim = world
            .resource_mut::<ReentrantState>()
            .claim
            .take()
            .expect("final claim");
        let identity = claim.entity();
        claim.try_release(world).expect("callback release succeeds");
        world.resource_mut::<ReentrantState>().released = true;
        assert!(
            world.get_entity(identity.entity()).is_ok(),
            "ordinary terminal is structurally retained"
        );
        if world.resource::<ReentrantState>().reregister {
            let retained = world.resource::<RetainedTerminal>().0.clone();
            let claim = retained
                .try_register(world)
                .expect("same-callback registration succeeds");
            assert_eq!(
                claim.entity(),
                identity,
                "same retained terminal, not a replacement node"
            );
            let mut state = world.resource_mut::<ReentrantState>();
            state.registered = true;
            state.claim = Some(claim);
        }
    }
}

fn reentrant_case(drain: Drain, reregister: bool, extra_claim: bool) {
    let mut app = app();
    app.init_resource::<ReentrantState>();
    let values = MutableVec::<i32>::builder().spawn(app.world_mut());
    let source = values.signal_vec().schedule::<Producer>().broadcast();
    let producer = source.clone().register(app.world_mut());
    let terminal = source.for_each(reentrant_callback).schedule::<Consumer>().broadcast();
    let claim = terminal.clone().register(app.world_mut());
    let identity = claim.entity();
    let sibling = extra_claim.then(|| terminal.clone().register(app.world_mut()));
    app.insert_resource(RetainedTerminal(terminal.clone()));
    app.world_mut().run_schedule(Consumer);
    assert_eq!(
        app.world().resource::<ReentrantState>().calls,
        ["[Replace { values: [] }]"]
    );
    app.insert_resource(ReentrantState {
        claim: Some(claim),
        armed: true,
        reregister,
        ..Default::default()
    });
    queue_pushes(&mut app, &values, &producer, drain, [1, 2, 3]);
    match drain {
        Drain::Scheduled => app.world_mut().run_schedule(Consumer),
        #[cfg(feature = "imperative-signal-evaluation")]
        Drain::Polled => {
            // Same-callback scalar output acceptance is deliberately not asserted here.
            drop(run_signal_subgraph_once(app.world_mut(), identity));
        }
    }
    let old_drain = core::mem::take(&mut app.world_mut().resource_mut::<ReentrantState>().calls);
    assert!(app.world().resource::<ReentrantState>().released);
    assert_eq!(app.world().resource::<ReentrantState>().registered, reregister);
    app.world_mut().run_schedule(Consumer);
    let next_drain = core::mem::take(&mut app.world_mut().resource_mut::<ReentrantState>().calls);
    if !reregister {
        let claim = terminal
            .clone()
            .try_register(app.world_mut())
            .expect("outside-callback reactivation");
        assert_eq!(claim.entity(), identity);
        app.world_mut().resource_mut::<ReentrantState>().claim = Some(claim);
        app.world_mut().run_schedule(Consumer);
        assert_eq!(
            app.world().resource::<ReentrantState>().calls,
            ["[Replace { values: [1, 2, 3] }]"],
            "fresh authoritative replay, not old B/C delta delivery"
        );
        app.world_mut().resource_mut::<ReentrantState>().calls.clear();
    }
    queue_pushes(&mut app, &values, &producer, drain, [4]);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<ReentrantState>().calls, ["[Push { value: 4 }]"]);
    app.world_mut().resource_mut::<ReentrantState>().calls.clear();
    app.world_mut().run_schedule(Consumer);
    assert!(app.world().resource::<ReentrantState>().calls.is_empty());
    println!(
        "{drain:?}, reregister={reregister}, extra_claim={extra_claim}: release=Ok, register={}, old drain={old_drain:?}, next drain={next_drain:?}",
        app.world().resource::<ReentrantState>().registered
    );
    app.world_mut()
        .resource_mut::<ReentrantState>()
        .claim
        .take()
        .expect("reactivated claim")
        .release(app.world_mut());
    if let Some(sibling) = sibling {
        sibling.release(app.world_mut());
    }
    producer.release(app.world_mut());
    let all_calls = old_drain.into_iter().chain(next_drain).collect::<Vec<_>>();
    // Scheduled registration recursion may consume fresh replay in the same schedule. Its
    // Replace batch is distinct from the invalidated Push suffix regardless of that timing.
    let expected = if extra_claim {
        vec!["[Push { value: 1 }]", "[Push { value: 2 }]", "[Push { value: 3 }]"]
    } else if reregister {
        vec!["[Push { value: 1 }]", "[Replace { values: [1, 2, 3] }]"]
    } else {
        vec!["[Push { value: 1 }]"]
    };
    assert_eq!(
        all_calls, expected,
        "final release invalidates old B/C pushes; reactivation delivers one distinct authoritative replay"
    );
}

#[test]
fn scheduled_final_release_reregister_invalidates_detached_suffix() {
    reentrant_case(Drain::Scheduled, true, false);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn polled_final_release_reregister_invalidates_detached_suffix() {
    reentrant_case(Drain::Polled, true, false);
}

#[test]
fn non_final_release_reregister_preserves_detached_suffix_control() {
    reentrant_case(Drain::Scheduled, true, true);
    #[cfg(feature = "imperative-signal-evaluation")]
    reentrant_case(Drain::Polled, true, true);
}

#[test]
fn final_release_without_same_callback_reactivation_control() {
    reentrant_case(Drain::Scheduled, false, false);
    #[cfg(feature = "imperative-signal-evaluation")]
    reentrant_case(Drain::Polled, false, false);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[derive(Resource, Default, Debug)]
struct DynamicObservations {
    manager_calls: usize,
    updates: Vec<i32>,

    delivered: Vec<Vec<i32>>,
}

#[cfg(feature = "imperative-signal-evaluation")]
struct EmissionOnly {
    ordinary_terminal: bool,
    reductions: alloc::sync::Arc<std::sync::Mutex<Vec<Vec<i32>>>>,
}

#[cfg(feature = "imperative-signal-evaluation")]
impl Signal for EmissionOnly {
    type Item = Vec<i32>;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self {
            ordinary_terminal,
            reductions,
        } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                |In(()), mut pending: ResMut<Pending>, mut observations: ResMut<DynamicObservations>| {
                    observations.manager_calls += 1;
                    let value = pending.0.take()?;
                    observations.updates.push(value);
                    // No selection is ever constructed, so subscriptions remain empty throughout.
                    Some(DynamicManagerUpdate::new(
                        Vec::<ScalarSlotCommand<BoxedReplayableSignal<i32>>>::new(),
                        DynamicEmissions::One(value),
                    ))
                },
                OutputDelivery::move_only(),
            )?;
            let manager = registrar.apply_schedule(manager, Producer)?;
            let output = registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .replayable_scalar_output(move |values| {
                        reductions.lock().expect("reducer log").push(values.clone());
                        if values.is_empty() { None } else { Some(values) }
                    })
                    .same_signal_identity(SameSignalIdentity::KeepCurrent),
            )?;
            // The bridge inherits Producer before this changes only the output's schedule.
            let output = registrar.apply_schedule(output, Consumer)?;
            if ordinary_terminal {
                let terminal = registrar.signal_entity(
                    output,
                    |In(values): In<Vec<i32>>, mut observations: ResMut<DynamicObservations>| {
                        observations.delivered.push(values.clone());
                        Some(values)
                    },
                    OutputDelivery::broadcast(),
                )?;
                registrar.apply_schedule(terminal, Delivery)
            } else {
                Ok(output)
            }
        })
    }
}

#[cfg(feature = "imperative-signal-evaluation")]
fn dynamic_case(release: bool, ordinary_terminal: bool) {
    let mut app = app();
    app.init_resource::<DynamicObservations>();
    let reductions = alloc::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let operator = EmissionOnly {
        ordinary_terminal,
        reductions: reductions.clone(),
    };
    let retained = if ordinary_terminal {
        operator.schedule::<Delivery>().broadcast()
    } else {
        operator.schedule::<Consumer>().broadcast()
    };
    let claim = retained.clone().register(app.world_mut());
    let identity = claim.entity();
    app.world_mut().resource_mut::<Pending>().0 = Some(1);
    app.world_mut().run_schedule(Producer);
    assert_eq!(app.world().resource::<DynamicObservations>().updates, [1]);
    assert!(app.world().resource::<DynamicObservations>().delivered.is_empty());
    assert!(
        reductions.lock().expect("reducer log").is_empty(),
        "manager processed before output schedule"
    );
    if release {
        claim
            .try_release(app.world_mut())
            .expect("release before output schedule");
        assert!(
            app.world().get_entity(identity.entity()).is_ok(),
            "retained ordinary terminal remains allocated"
        );
        let calls = app.world().resource::<DynamicObservations>().manager_calls;
        app.world_mut().run_schedule(Producer);
        assert_eq!(
            app.world().resource::<DynamicObservations>().manager_calls,
            calls + usize::from(!ordinary_terminal),
            "raw output beneath ordinary retainer must deactivate its manager; direct replay retention intentionally keeps it active"
        );
    } else {
        // Keep the original active claim, without introducing replay-maintenance ownership.
        app.insert_resource(DynamicClaim(Some(claim)));
    }
    if release && ordinary_terminal {
        app.world_mut().run_schedule(Consumer);
        app.world_mut().run_schedule(Delivery);
        assert!(
            reductions.lock().expect("reducer log").is_empty(),
            "inactive raw output must not execute"
        );
    }
    let claim = retained
        .clone()
        .try_register(app.world_mut())
        .expect("reactivate retained terminal");
    assert_eq!(claim.entity(), identity);
    if release && ordinary_terminal {
        app.world_mut().run_schedule(Consumer);
        app.world_mut().run_schedule(Delivery);
        assert!(
            reductions.lock().expect("reducer log").is_empty(),
            "release canceled the old graph wake; only a fresh manager emission can wake the output"
        );
        assert!(app.world().resource::<DynamicObservations>().delivered.is_empty());
    }
    app.world_mut().resource_mut::<Pending>().0 = Some(2);
    app.world_mut().run_schedule(Producer);
    assert_eq!(app.world().resource::<DynamicObservations>().updates, [1, 2]);
    app.world_mut().run_schedule(Consumer);
    let output = run_signal_subgraph_once(app.world_mut(), identity).and_then(downcast_signal_value::<Vec<i32>>);
    let delivered = app.world().resource::<DynamicObservations>().delivered.clone();
    let reductions = reductions.lock().expect("reducer log").clone();
    println!(
        "dynamic release={release}, ordinary_terminal={ordinary_terminal}: output={output:?}, delivered={delivered:?}, reductions={reductions:?}, observations={:?}",
        app.world().resource::<DynamicObservations>()
    );
    claim.release(app.world_mut());
    if !release {
        app.world_mut()
            .resource_mut::<DynamicClaim>()
            .0
            .take()
            .expect("control claim")
            .release(app.world_mut());
    }
    if ordinary_terminal {
        let expected = if release { vec![2] } else { vec![1, 2] };
        assert_eq!(
            output,
            Some(expected.clone()),
            "new wake must not drain an emission from the previous activation"
        );
        assert_eq!(delivered, [expected]);
    } else {
        // Polling is not replay inspection; Consumer already accepted the scalar output.
        assert_eq!(output, None);
        assert_eq!(
            reductions
                .into_iter()
                .filter(|values| !values.is_empty())
                .collect::<Vec<_>>(),
            [vec![1, 2]],
            "non-final direct replay retention correctly preserves both emissions"
        );
    }
}

#[cfg(feature = "imperative-signal-evaluation")]
#[derive(Resource)]
struct DynamicClaim(Option<SignalClaim>);

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn emission_only_dynamic_queue_is_invalidated_beneath_retained_ordinary_terminal() {
    dynamic_case(true, true);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn emission_only_dynamic_active_queue_control() {
    dynamic_case(false, true);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn direct_dynamic_replay_retention_is_not_final_release_control() {
    dynamic_case(true, false);
}

#[derive(Resource, Default)]
struct BoundaryState {
    claim: Option<SignalClaim>,
    deliver_before_release: bool,
    calls: Vec<i32>,
    delivered: Vec<String>,
    released: bool,
}

struct BoundaryCollection(Broadcast<i32>);

impl SignalVec for BoundaryCollection {
    type Item = i32;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
        VecSignalMaterializer::new(move |registrar| {
            let source = registrar.materialize_signal(Box::new(self.0).into_signal_materializer())?;
            registrar.replayable_signal_vec_entity(
                source,
                |In(value): In<i32>, world: &mut World| {
                    world.resource_mut::<BoundaryState>().calls.push(value);
                    if value == 2 {
                        if world.resource::<BoundaryState>().deliver_before_release {
                            // Delivery runs only the ordinary observer, not this in-flight
                            // producer. Consume A before release can cancel the observer's pending
                            // inputs.
                            world.run_schedule(Delivery);
                        }
                        let claim = world
                            .resource_mut::<BoundaryState>()
                            .claim
                            .take()
                            .expect("only remaining terminal owner");
                        claim.try_release(world).expect("B releases final owner");
                        world.resource_mut::<BoundaryState>().released = true;
                    }
                    Some(vec![VecDiff::Replace { values: vec![value] }])
                },
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        })
    }
}

fn boundary_case(drain: Drain, deliver_before_release: bool) {
    let mut app = app();
    app.insert_resource(BoundaryState {
        deliver_before_release,
        ..Default::default()
    });
    let source = signal::from_system(pending).schedule::<Producer>().broadcast();
    let producer = source.clone().register(app.world_mut());
    let collection = BoundaryCollection(source).schedule::<Consumer>().broadcast();
    let collection_claim = collection.clone().register(app.world_mut());
    let identity = collection_claim.entity();
    // Consume the last collection handle into an ordinary terminal. Retaining the collection
    // directly would add replay-maintenance ownership and make B's release non-final.
    let terminal = collection
        .for_each(|In(batch): In<Vec<VecDiff<i32>>>, mut state: ResMut<BoundaryState>| {
            state.delivered.push(format!("{batch:?}"));
        })
        .schedule::<Delivery>()
        .broadcast();
    let claim = terminal.clone().register(app.world_mut());
    let terminal_identity = claim.entity();
    app.update();
    collection_claim
        .try_release(app.world_mut())
        .expect("release temporary collection owner after retainer cleanup");
    app.world_mut().resource_mut::<BoundaryState>().claim = Some(claim);
    queue(&mut app, &producer, drain, [1, 2, 3]);
    let output = match drain {
        Drain::Scheduled => {
            app.world_mut().run_schedule(Consumer);
            None::<Vec<VecDiff<i32>>>
        }
        #[cfg(feature = "imperative-signal-evaluation")]
        Drain::Polled => {
            run_signal_subgraph_once(app.world_mut(), identity).and_then(downcast_signal_value::<Vec<VecDiff<i32>>>)
        }
    };
    let _ = &output;
    assert!(app.world().resource::<BoundaryState>().released);
    assert_eq!(
        app.world().resource::<BoundaryState>().calls,
        [1, 2],
        "C never executes after B deactivates the producer"
    );
    assert!(
        app.world().get_entity(identity.entity()).is_ok(),
        "raw replay output is structurally retained"
    );
    match drain {
        Drain::Scheduled => {}
        #[cfg(feature = "imperative-signal-evaluation")]
        Drain::Polled => {
            assert_eq!(
                format!("{:?}", output.expect("A was accepted before B")),
                "[Replace { values: [1] }]"
            );
            assert!(
                run_signal_subgraph_once(app.world_mut(), identity).is_none(),
                "inactive output cannot run"
            );
        }
    }
    app.world_mut().run_schedule(Delivery);
    let delivered_before_reactivation = core::mem::take(&mut app.world_mut().resource_mut::<BoundaryState>().delivered);
    assert_eq!(
        delivered_before_reactivation,
        if deliver_before_release {
            vec!["[Replace { values: [1] }]"]
        } else {
            vec![]
        },
        "only accepted A may have been delivered; final release cancels any still-pending observer input"
    );
    let claim = terminal
        .clone()
        .try_register(app.world_mut())
        .expect("reactivate retained ordinary observer");
    assert_eq!(claim.entity(), terminal_identity);
    app.world_mut().run_schedule(Delivery);
    assert_eq!(
        app.world().resource::<BoundaryState>().delivered,
        ["[Replace { values: [1] }]"],
        "only A entered replay and is delivered to the new activation"
    );
    app.world_mut().run_schedule(Consumer);
    app.world_mut().run_schedule(Delivery);
    assert_eq!(app.world().resource::<BoundaryState>().calls, [1, 2]);
    assert_eq!(app.world().resource::<BoundaryState>().delivered.len(), 1);
    println!(
        "{drain:?} acceptance boundary: callbacks=A,B; B release=Ok; C canceled; before reactivation={delivered_before_reactivation:?}; replay/delivery=[A]"
    );
    claim.release(app.world_mut());
    producer.release(app.world_mut());
}

#[test]
fn scheduled_collection_acceptance_precedes_later_final_release_control() {
    boundary_case(Drain::Scheduled, false);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn polled_collection_returns_only_pre_release_accepted_output_control() {
    boundary_case(Drain::Polled, false);
}

#[test]
fn collection_accepted_output_is_delivered_before_later_release_control() {
    boundary_case(Drain::Scheduled, true);
    #[cfg(feature = "imperative-signal-evaluation")]
    boundary_case(Drain::Polled, true);
}
