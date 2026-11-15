//! Queued inputs live until consumption or release, rather than until the end of an app update.

use bevy_app::{App, PostUpdate, Update};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use jonmo::{MapSignalMaterializer, OutputDelivery, ScalarSignalMaterializer, VecSignalMaterializer, prelude::*};

#[derive(Resource, Default, Debug)]
struct Observations {
    scalars: Vec<i32>,
    optional_scalars: Vec<Option<i32>>,
    produced: Vec<i32>,
    snapshot_batches: Vec<Vec<VecDiff<i32>>>,
    map_batches: Vec<Vec<MapDiff<i32, i32>>>,
}

#[derive(Resource, Default)]
struct Claims(Vec<SignalClaim>);

fn app(plugin: JonmoPlugin) -> App {
    let mut app = App::new();
    app.add_plugins(plugin);
    app.init_resource::<Observations>();
    app.init_resource::<Claims>();
    app.init_resource::<PendingScalar>();
    app
}

fn pump(app: &mut App, frames: usize) {
    for _ in 0..frames {
        app.update();
    }
}

fn release_claims(app: &mut App) {
    let claims = core::mem::take(&mut app.world_mut().resource_mut::<Claims>().0);
    for claim in claims {
        claim.release(app.world_mut());
    }
}

fn observe_scalar(In(value): In<i32>, mut seen: ResMut<Observations>) {
    seen.scalars.push(value);
}

fn observe_snapshot(In(diffs): In<Vec<VecDiff<i32>>>, mut seen: ResMut<Observations>) {
    seen.snapshot_batches.push(diffs);
}

#[test]
fn post_update_registration_must_deliver_ready_snapshot_next_update() {
    let mut control = app(JonmoPlugin::new::<Update>());
    let control_values = MutableVec::builder().values([41]).spawn(control.world_mut());
    let claim = control_values
        .signal_vec()
        .for_each(observe_snapshot)
        .register(control.world_mut());
    control.update();
    let control_batches = &control.world().resource::<Observations>().snapshot_batches;
    println!("registered before Update control: {control_batches:?}");
    assert!(
        matches!(control_batches.as_slice(), [diffs] if matches!(diffs.as_slice(), [VecDiff::Replace { values }] if values == &[41]))
    );
    claim.release(control.world_mut());

    let mut app = app(JonmoPlugin::new::<Update>());
    let values = MutableVec::builder().values([41]).spawn(app.world_mut());
    app.add_systems(PostUpdate, {
        let values = values.clone();
        move |world: &mut World, mut registered: Local<bool>| {
            if !*registered {
                *registered = true;
                let claim = values.signal_vec().for_each(observe_snapshot).register(world);
                world.resource_mut::<Claims>().0.push(claim);
            }
        }
    });
    for frame in 1..=3 {
        app.update();
        println!(
            "late registration frame {frame}: {:?}",
            app.world().resource::<Observations>().snapshot_batches
        );
    }
    let initial_batches = app.world().resource::<Observations>().snapshot_batches.clone();
    values.write(app.world_mut()).push(42);
    app.update();
    println!(
        "late registration after Push42: {:?}",
        app.world().resource::<Observations>().snapshot_batches
    );
    assert!(
        app.world()
            .resource::<Observations>()
            .snapshot_batches
            .iter()
            .any(|diffs| matches!(diffs.as_slice(), [VecDiff::Push { value: 42 }])),
        "the late observer is active"
    );
    let claims = core::mem::take(&mut app.world_mut().resource_mut::<Claims>().0);
    for claim in claims {
        claim.release(app.world_mut());
    }
    assert!(
        matches!(initial_batches.as_slice(), [diffs] if matches!(diffs.as_slice(), [VecDiff::Replace { values }] if values == &[41])),
        "ready snapshot must survive until the consumer schedule: actual {initial_batches:?}"
    );
}

fn register_after_update(app: &mut App, register: impl Fn(&mut World) -> SignalClaim + Send + Sync + 'static) {
    app.add_systems(PostUpdate, move |world: &mut World, mut registered: Local<bool>| {
        if !core::mem::replace(&mut *registered, true) {
            let claim = register(world);
            world.resource_mut::<Claims>().0.push(claim);
        }
    });
}

struct ReadyScalar<T>(T);

impl<T: Clone + Send + Sync + 'static> Signal for ReadyScalar<T> {
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<T> {
        ScalarSignalMaterializer::new(move |registrar| {
            registrar.replayable_signal_source(self.0, |In(())| None::<T>, OutputDelivery::broadcast())
        })
    }
}

#[test]
fn external_ready_scalar_registered_after_update_replays_once_next_update() {
    let mut app = app(JonmoPlugin::new::<Update>());
    register_after_update(&mut app, |world| ReadyScalar(41).map(observe_scalar).register(world));
    app.update();
    assert!(app.world().resource::<Observations>().scalars.is_empty());
    app.update();
    assert_eq!(app.world().resource::<Observations>().scalars, [41]);
    pump(&mut app, 3);
    assert_eq!(
        app.world().resource::<Observations>().scalars,
        [41],
        "no replay repeats"
    );
    release_claims(&mut app);
}

#[test]
fn external_ready_none_registered_after_update_is_an_authoritative_value() {
    let mut app = app(JonmoPlugin::new::<Update>());
    register_after_update(&mut app, |world| {
        ReadyScalar(None::<i32>)
            .map(|In(value): In<Option<i32>>, mut seen: ResMut<Observations>| {
                seen.optional_scalars.push(value);
            })
            .register(world)
    });
    app.update();
    assert!(app.world().resource::<Observations>().optional_scalars.is_empty());
    app.update();
    assert_eq!(app.world().resource::<Observations>().optional_scalars, [None]);
    pump(&mut app, 3);
    assert_eq!(app.world().resource::<Observations>().optional_scalars, [None]);
    release_claims(&mut app);
}

struct EmptyVector;

impl SignalVec for EmptyVector {
    type Item = i32;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
        VecSignalMaterializer::new(|registrar| {
            registrar.replayable_signal_vec_source(
                Vec::new(),
                |In(())| None::<Vec<VecDiff<i32>>>,
                OutputDelivery::broadcast(),
            )
        })
    }
}

struct EmptyMap;

impl SignalMap for EmptyMap {
    type Key = i32;
    type Value = i32;

    fn into_signal_map_materializer(self: Box<Self>) -> MapSignalMaterializer<i32, i32> {
        MapSignalMaterializer::new(|registrar| {
            registrar.replayable_signal_map_source(
                Default::default(),
                |In(())| None::<Vec<MapDiff<i32, i32>>>,
                OutputDelivery::broadcast(),
            )
        })
    }
}

#[test]
fn external_empty_vector_registered_after_update_delivers_one_authoritative_replace() {
    let mut app = app(JonmoPlugin::new::<Update>());
    register_after_update(&mut app, |world| EmptyVector.for_each(observe_snapshot).register(world));
    app.update();
    assert!(app.world().resource::<Observations>().snapshot_batches.is_empty());
    app.update();
    let batches = &app.world().resource::<Observations>().snapshot_batches;
    assert!(
        matches!(batches.as_slice(), [diffs] if matches!(diffs.as_slice(), [VecDiff::Replace { values }] if values.is_empty())),
        "empty is ready, not warming: {batches:?}"
    );
    pump(&mut app, 3);
    assert_eq!(app.world().resource::<Observations>().snapshot_batches.len(), 1);
    release_claims(&mut app);
}

#[test]
fn external_empty_map_registered_after_update_delivers_one_authoritative_replace() {
    let mut app = app(JonmoPlugin::new::<Update>());
    register_after_update(&mut app, |world| {
        EmptyMap
            .for_each(
                |In(diffs): In<Vec<MapDiff<i32, i32>>>, mut seen: ResMut<Observations>| {
                    seen.map_batches.push(diffs);
                },
            )
            .register(world)
    });
    app.update();
    assert!(app.world().resource::<Observations>().map_batches.is_empty());
    app.update();
    let batches = &app.world().resource::<Observations>().map_batches;
    assert!(
        matches!(batches.as_slice(), [diffs] if matches!(diffs.as_slice(), [MapDiff::Replace { entries }] if entries.is_empty())),
        "empty is ready, not warming: {batches:?}"
    );
    pump(&mut app, 3);
    assert_eq!(app.world().resource::<Observations>().map_batches.len(), 1);
    release_claims(&mut app);
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct Producer;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct ProducerFirst;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct ProducerSecond;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct Consumer;

#[derive(Resource, Default)]
struct PendingScalar(Option<i32>);

fn emit_pending(In(()): In<()>, mut pending: ResMut<PendingScalar>, mut seen: ResMut<Observations>) -> Option<i32> {
    let value = pending.0.take()?;
    seen.produced.push(value);
    Some(value)
}

#[test]
fn registered_producer_in_later_schedule_delivers_to_consumer_next_update() {
    let mut app = app(JonmoPlugin::new::<Update>().with_schedule::<PostUpdate>());
    let source = signal::from_system(emit_pending).schedule::<PostUpdate>().broadcast();
    let producer = source.clone().register(app.world_mut());
    let consumer = source
        .map(observe_scalar)
        .schedule::<Update>()
        .register(app.world_mut());
    app.world_mut().resource_mut::<PendingScalar>().0 = Some(7);
    app.update();
    assert_eq!(app.world().resource::<Observations>().produced, [7]);
    assert!(app.world().resource::<Observations>().scalars.is_empty());
    app.update();
    assert_eq!(app.world().resource::<Observations>().scalars, [7]);
    pump(&mut app, 3);
    assert_eq!(app.world().resource::<Observations>().produced, [7]);
    assert_eq!(app.world().resource::<Observations>().scalars, [7]);
    consumer.release(app.world_mut());
    producer.release(app.world_mut());
}

#[test]
fn ordinary_inputs_survive_skipped_consumer_schedule_executions() {
    let mut app = app(JonmoPlugin::new::<Update>().with_schedule::<Consumer>());
    let source = signal::from_system(emit_pending).schedule::<Update>().broadcast();
    let producer = source.clone().register(app.world_mut());
    let consumer = source
        .map(observe_scalar)
        .schedule::<Consumer>()
        .register(app.world_mut());
    for value in [1, 2, 3] {
        app.world_mut().resource_mut::<PendingScalar>().0 = Some(value);
        app.update();
        assert!(app.world().resource::<Observations>().scalars.is_empty());
    }
    assert_eq!(app.world().resource::<Observations>().produced, [1, 2, 3]);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [1, 2, 3]);
    app.world_mut().run_schedule(Consumer);
    pump(&mut app, 2);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [1, 2, 3]);
    consumer.release(app.world_mut());
    producer.release(app.world_mut());
}

fn manual_app() -> App {
    app(JonmoPlugin::new::<Producer>()
        .with_schedule::<ProducerFirst>()
        .with_schedule::<ProducerSecond>()
        .with_schedule::<Consumer>())
}

fn counter_in<S: ScheduleLabel + Clone + Default>(start: i32) -> Broadcast<i32> {
    signal::from_system(move |In(()), mut offset: Local<i32>, mut seen: ResMut<Observations>| {
        let value = start + *offset;
        *offset += 1;
        seen.produced.push(value);
        Some(value)
    })
    .schedule::<S>()
    .broadcast()
}

fn counter(start: i32) -> Broadcast<i32> {
    counter_in::<Producer>(start)
}

fn step_schedule<S: ScheduleLabel>(app: &mut App, schedule: S, expected: i32) {
    let before = app.world().resource::<Observations>().produced.len();
    app.world_mut().run_schedule(schedule);
    assert_eq!(
        &app.world().resource::<Observations>().produced[before..],
        &[expected],
        "explicit producer schedule execution must accept exactly the requested event"
    );
}

fn ordinary_source_order_case(cross_frame: bool) {
    let mut app = manual_app();
    let first = counter_in::<ProducerFirst>(10);
    let second = counter_in::<ProducerSecond>(20);
    let first_claim = first.clone().register(app.world_mut());
    let second_claim = second.clone().register(app.world_mut());
    let consumer = ScalarSignalMaterializer::<()>::new(move |registrar| {
        let first = registrar.materialize_signal(Box::new(first).into_signal_materializer())?;
        let second = registrar.materialize_signal(Box::new(second).into_signal_materializer())?;
        let consumer = registrar.signal_entity(
            [first, second],
            |In(value): In<i32>, mut seen: ResMut<Observations>| {
                seen.scalars.push(value);
                None::<()>
            },
            OutputDelivery::broadcast(),
        )?;
        registrar.apply_schedule(consumer, Consumer)
    })
    .register(app.world_mut());
    for (first_slot, value) in [(true, 10), (false, 20), (true, 11)] {
        if first_slot {
            step_schedule(&mut app, ProducerFirst, value);
        } else {
            step_schedule(&mut app, ProducerSecond, value);
        }
        if cross_frame {
            app.update();
        }
        assert!(app.world().resource::<Observations>().scalars.is_empty());
    }
    assert_eq!(app.world().resource::<Observations>().produced, [10, 20, 11]);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [10, 20, 11]);
    app.world_mut().run_schedule(Consumer);
    pump(&mut app, 2);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [10, 20, 11]);
    consumer.release(app.world_mut());
    first_claim.release(app.world_mut());
    second_claim.release(app.world_mut());
}

#[test]
fn ordinary_inputs_preserve_cross_source_enqueue_order_across_updates() {
    ordinary_source_order_case(true);
}

#[test]
fn ordinary_inputs_preserve_cross_source_enqueue_order_same_frame_control() {
    ordinary_source_order_case(false);
}

struct UpdatingScalar(i32);

impl Signal for UpdatingScalar {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        ScalarSignalMaterializer::new(move |registrar| {
            registrar.replayable_signal_source(self.0, emit_pending, OutputDelivery::broadcast())
        })
    }
}

fn scalar_replay_suffix_case(cross_frame: bool) {
    let mut app = manual_app();
    let source = UpdatingScalar(41).schedule::<Producer>().broadcast();
    let producer = source.clone().register(app.world_mut());
    let consumer = source
        .map(observe_scalar)
        .schedule::<Consumer>()
        .register(app.world_mut());
    for value in [42, 43] {
        app.world_mut().resource_mut::<PendingScalar>().0 = Some(value);
        step_schedule(&mut app, Producer, value);
        if cross_frame {
            app.update();
        }
        assert!(app.world().resource::<Observations>().scalars.is_empty());
    }
    assert_eq!(app.world().resource::<Observations>().produced, [42, 43]);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [41, 42, 43]);
    app.world_mut().run_schedule(Consumer);
    pump(&mut app, 2);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [41, 42, 43]);
    consumer.release(app.world_mut());
    producer.release(app.world_mut());
}

#[test]
fn scalar_initial_replay_then_suffix_survive_updates_before_consumption() {
    scalar_replay_suffix_case(true);
}

#[test]
fn scalar_initial_replay_then_suffix_same_frame_control() {
    scalar_replay_suffix_case(false);
}

#[test]
fn release_before_consumption_discards_ordinary_inputs_before_retained_reactivation() {
    let mut app = manual_app();
    let source = counter(1);
    let producer = source.clone().register(app.world_mut());
    let retained = source.map(observe_scalar).schedule::<Consumer>().broadcast();
    let consumer = retained.clone().register(app.world_mut());
    let identity = consumer.entity();
    step_schedule(&mut app, Producer, 1);
    consumer.try_release(app.world_mut()).expect("release queued consumer");
    assert!(app.world().get_entity(identity.entity()).is_ok());
    app.world_mut().run_schedule(Consumer);
    assert!(app.world().resource::<Observations>().scalars.is_empty());
    step_schedule(&mut app, Producer, 2);
    app.world_mut().run_schedule(Consumer);
    assert!(app.world().resource::<Observations>().scalars.is_empty());
    // Reactivate before Last can discard anything: release itself must invalidate old input.
    let consumer = retained
        .clone()
        .try_register(app.world_mut())
        .expect("reactivate retained consumer");
    assert_eq!(consumer.entity(), identity);
    app.world_mut().run_schedule(Consumer);
    assert!(app.world().resource::<Observations>().scalars.is_empty());
    step_schedule(&mut app, Producer, 3);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [3]);
    pump(&mut app, 2);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [3]);
    consumer.release(app.world_mut());
    producer.release(app.world_mut());
}

#[test]
fn release_before_consumption_reactivates_retained_scalar_with_only_current_replay() {
    let mut app = manual_app();
    let source = UpdatingScalar(41).schedule::<Producer>().broadcast();
    let producer = source.clone().register(app.world_mut());
    let retained = source.map(observe_scalar).schedule::<Consumer>().broadcast();
    let consumer = retained.clone().register(app.world_mut());
    let identity = consumer.entity();
    app.world_mut().resource_mut::<PendingScalar>().0 = Some(42);
    step_schedule(&mut app, Producer, 42);
    consumer
        .try_release(app.world_mut())
        .expect("release queued replay and suffix");
    assert!(app.world().get_entity(identity.entity()).is_ok());
    app.world_mut().run_schedule(Consumer);
    assert!(app.world().resource::<Observations>().scalars.is_empty());
    app.world_mut().resource_mut::<PendingScalar>().0 = Some(43);
    step_schedule(&mut app, Producer, 43);
    app.world_mut().run_schedule(Consumer);
    assert!(app.world().resource::<Observations>().scalars.is_empty());
    let consumer = retained
        .clone()
        .try_register(app.world_mut())
        .expect("reactivate retained replay consumer");
    assert_eq!(consumer.entity(), identity);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [43]);
    pump(&mut app, 2);
    app.world_mut().run_schedule(Consumer);
    assert_eq!(app.world().resource::<Observations>().scalars, [43]);
    consumer.release(app.world_mut());
    producer.release(app.world_mut());
}

mod remaining_reentrancy {
    use super::*;

    #[derive(Resource)]
    struct RetainedOrdinary(Broadcast<i32>);

    #[derive(Resource, Default)]
    struct ReleaseState {
        claim: Option<SignalClaim>,
        armed: bool,
        succeeded: bool,
        events: Vec<String>,
    }

    fn release_and_register(world: &mut World) {
        let claim = world
            .resource_mut::<ReleaseState>()
            .claim
            .take()
            .expect("initial claim");
        let identity = claim.entity();
        let retained = world.resource::<RetainedOrdinary>().0.clone();
        claim.try_release(world).expect("release should succeed");
        world.resource_mut::<ReleaseState>().events.push("release: Ok".into());
        assert!(
            world.get_entity(identity.entity()).is_ok(),
            "retained identity must still exist"
        );
        match retained.try_register(world) {
            Ok(claim) => {
                assert_eq!(claim.entity(), identity, "registration must reuse retained identity");
                let mut state = world.resource_mut::<ReleaseState>();
                state.events.push("same-callback register: Ok, same identity".into());
                state.succeeded = true;
                state.claim = Some(claim);
            }
            Err(error) => world
                .resource_mut::<ReleaseState>()
                .events
                .push(format!("same-callback register: Err({error:?})")),
        }
    }

    fn release_case(label: &str, outside_processing: bool, extra_claim: bool) -> bool {
        let mut app = app(JonmoPlugin::new::<Update>());
        app.init_resource::<ReleaseState>();
        let retained = signal::from_system(|In(()), world: &mut World| {
            world
                .resource_mut::<ReleaseState>()
                .events
                .push("ordinary root callback".into());
            if core::mem::take(&mut world.resource_mut::<ReleaseState>().armed) {
                release_and_register(world);
            }
            Some(7i32)
        })
        .broadcast();
        let claim = retained.clone().register(app.world_mut());
        let identity = claim.entity();
        let sibling = extra_claim.then(|| retained.clone().register(app.world_mut()));
        app.insert_resource(RetainedOrdinary(retained.clone()));
        app.world_mut().resource_mut::<ReleaseState>().claim = Some(claim);
        app.world_mut().resource_mut::<ReleaseState>().armed = !outside_processing;
        if outside_processing {
            release_and_register(app.world_mut());
        }
        app.update();
        let succeeded = app.world().resource::<ReleaseState>().succeeded;
        println!(
            "{label}: identity={identity:?}, events={:?}",
            app.world().resource::<ReleaseState>().events
        );
        if !succeeded {
            let retry = retained
                .clone()
                .try_register(app.world_mut())
                .expect("retry after graph processing must succeed");
            assert_eq!(retry.entity(), identity);
            println!("{label}: after-processing retry=Ok, same identity");
            app.world_mut().resource_mut::<ReleaseState>().claim = Some(retry);
        }
        let calls_before = app
            .world()
            .resource::<ReleaseState>()
            .events
            .iter()
            .filter(|event| event.as_str() == "ordinary root callback")
            .count();
        app.update();
        let calls_after = app
            .world()
            .resource::<ReleaseState>()
            .events
            .iter()
            .filter(|event| event.as_str() == "ordinary root callback")
            .count();
        assert_eq!(
            calls_after,
            calls_before + 1,
            "reactivated ordinary root runs exactly once next frame"
        );
        let claim = app
            .world_mut()
            .resource_mut::<ReleaseState>()
            .claim
            .take()
            .expect("live final claim");
        claim.release(app.world_mut());
        if let Some(sibling) = sibling {
            sibling.release(app.world_mut());
        }
        succeeded
    }

    #[test]
    fn release_then_same_callback_register_retained_ordinary() {
        assert!(release_case("outside-processing final-release control", true, false));
        assert!(release_case("same-callback non-final-release control", false, true));
        let succeeded = release_case("same-callback final release", false, false);
        assert!(
            succeeded,
            "retained ordinary signal should support immediate reactivation inside its callback"
        );
    }
}
