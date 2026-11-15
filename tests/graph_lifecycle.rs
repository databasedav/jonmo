//! Regression coverage for immediate graph ownership transitions and reentrant execution.

extern crate alloc;

use alloc::{collections::VecDeque, sync::Arc};
use bevy_app::{App, Update};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
#[cfg(feature = "imperative-signal-evaluation")]
use jonmo::graph::{downcast_signal_value, run_signal_subgraph_once};
use jonmo::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery, ReplayInitialization,
    SameSignalIdentity, ScalarSignalMaterializer, ScalarSlotCommand, prelude::*,
};
use std::sync::Mutex;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
struct Producer;

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::new::<Update>().with_schedule::<Producer>());
    app.update();
    app
}

fn pump(app: &mut App, frames: usize) {
    for _ in 0..frames {
        app.update();
    }
}

struct Deep(usize);

impl Signal for Deep {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        ScalarSignalMaterializer::new(move |registrar| {
            let mut selected =
                registrar.replayable_signal_source(41, |In(())| None::<i32>, OutputDelivery::broadcast())?;
            for _ in 0..self.0 {
                selected = registrar.replayable_signal_entity(
                    selected,
                    |In(value): In<i32>| Some(value),
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )?;
            }
            Ok(selected)
        })
    }
}

struct Selections {
    signals: Vec<BoxedReplayableSignal<i32>>,
    errors: Arc<Mutex<Vec<String>>>,
}

impl Signal for Selections {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        let Self { signals, errors } = *self;
        let mut signals = VecDeque::from(signals);
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| {
                    signals.pop_front().map(|signal| {
                        DynamicManagerUpdate::new([ScalarSlotCommand::Select { signal }], DynamicEmissions::<i32>::None)
                    })
                },
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value))
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent)
                    .on_candidate_error(move |error| {
                        errors.lock().expect("error log").push(format!("{error:?}"));
                    }),
            )
        })
    }
}

#[derive(Resource, Default)]
struct Outputs(Vec<i32>);

#[test]
fn settled_deep_dynamic_static_descendant_must_release_immediately() {
    let mut app = app();
    app.init_resource::<Outputs>();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let dynamic = Selections {
        signals: vec![Deep(8).cache().boxed_replayable()],
        errors: errors.clone(),
    }
    .cache()
    .broadcast();
    let owner = dynamic.clone().register(app.world_mut());
    let observer = dynamic
        .clone()
        .map(|In(value): In<i32>, mut outputs: ResMut<Outputs>| outputs.0.push(value))
        .register(app.world_mut());
    pump(&mut app, 16);
    assert_eq!(app.world().resource::<Outputs>().0, [41]);
    assert!(errors.lock().expect("error log").is_empty());
    observer.try_release(app.world_mut()).expect("settled observer release");
    let descendant = dynamic.clone().map(|In(value): In<i32>| value).broadcast();
    let claim = descendant
        .clone()
        .try_register(app.world_mut())
        .expect("late descendant");
    claim
        .try_release(app.world_mut())
        .expect("immediate descendant release");
    let claim = descendant
        .clone()
        .try_register(app.world_mut())
        .expect("immediate reactivation");
    let further = descendant
        .map(|In(value): In<i32>| value)
        .try_register(app.world_mut())
        .expect("further descendant before scheduler repair");
    further.try_release(app.world_mut()).expect("further release");
    claim.try_release(app.world_mut()).expect("reactivated release");
    owner.try_release(app.world_mut()).expect("owner release");
}

#[test]
fn settled_deep_static_descendant_immediate_release_control() {
    let mut app = app();
    let selected = Deep(8).cache().broadcast();
    let owner = selected.clone().register(app.world_mut());
    pump(&mut app, 16);
    let descendant = selected.map(|In(value): In<i32>| value).register(app.world_mut());
    descendant
        .try_release(app.world_mut())
        .expect("static descendant release");
    owner.try_release(app.world_mut()).expect("owner release");
}

#[derive(Resource)]
struct RetainedOrdinary(Broadcast<i32>);

#[derive(Resource, Default)]
struct ReleaseState {
    claim: Option<SignalClaim>,
    armed: bool,
    calls: usize,
}

fn release_and_register(world: &mut World) {
    let claim = world
        .resource_mut::<ReleaseState>()
        .claim
        .take()
        .expect("initial claim");
    let identity = claim.entity();
    let retained = world.resource::<RetainedOrdinary>().0.clone();
    claim.try_release(world).expect("release");
    assert!(world.get_entity(identity.entity()).is_ok());
    let claim = retained.try_register(world).expect("same-callback registration");
    assert_eq!(claim.entity(), identity);
    world.resource_mut::<ReleaseState>().claim = Some(claim);
}

fn release_case(outside_processing: bool, extra_claim: bool) {
    let mut app = app();
    app.init_resource::<ReleaseState>();
    let retained = signal::from_system(|In(()), world: &mut World| {
        world.resource_mut::<ReleaseState>().calls += 1;
        if core::mem::take(&mut world.resource_mut::<ReleaseState>().armed) {
            release_and_register(world);
        }
        Some(7i32)
    })
    .broadcast();
    let claim = retained.clone().register(app.world_mut());
    let sibling = extra_claim.then(|| retained.clone().register(app.world_mut()));
    app.insert_resource(RetainedOrdinary(retained));
    app.world_mut().resource_mut::<ReleaseState>().claim = Some(claim);
    app.world_mut().resource_mut::<ReleaseState>().armed = !outside_processing;
    if outside_processing {
        release_and_register(app.world_mut());
    }
    app.update();
    assert_eq!(app.world().resource::<ReleaseState>().calls, 1);
    app.update();
    assert_eq!(
        app.world().resource::<ReleaseState>().calls,
        2,
        "reactivation retains exactly one schedule membership"
    );
    let claim = app
        .world_mut()
        .resource_mut::<ReleaseState>()
        .claim
        .take()
        .expect("final claim");
    claim.try_release(app.world_mut()).expect("final release");
    if let Some(sibling) = sibling {
        sibling.try_release(app.world_mut()).expect("sibling release");
    }
}

#[test]
fn release_then_same_callback_register_retained_ordinary() {
    release_case(false, false);
}

#[test]
fn outside_processing_reactivation_control() {
    release_case(true, false);
}

#[test]
fn same_callback_non_final_release_control() {
    release_case(false, true);
}

#[derive(Resource, Default)]
struct SelfClaim(Option<SignalClaim>);

#[derive(Clone, Copy)]
enum BufferedStep {
    Scheduled,
    #[cfg(feature = "imperative-signal-evaluation")]
    Imperative,
}

fn buffered_release_case(step: BufferedStep, retained: bool, release: bool) {
    let mut app = app();
    app.init_resource::<SelfClaim>();
    app.init_resource::<Outputs>();
    let values = MutableVec::<i32>::builder().spawn(app.world_mut());
    let source = values.signal_vec().schedule::<Producer>().register(app.world_mut());
    let terminal = values
        .signal_vec()
        .for_each(|In(diffs): In<Vec<VecDiff<i32>>>, world: &mut World| {
            for diff in diffs {
                if let VecDiff::Push { value } = diff {
                    world.resource_mut::<Outputs>().0.push(value);
                    let claim = world.resource_mut::<SelfClaim>().0.take();
                    if let Some(claim) = claim {
                        claim.try_release(world).expect("self release");
                    }
                }
            }
        })
        .schedule::<Update>()
        .broadcast();
    let claim = terminal.clone().register(app.world_mut());
    let identity = claim.entity();
    app.world_mut().run_schedule(Producer);
    app.update();
    let retained_terminal = retained.then_some(terminal);
    let remaining_claim = if release {
        app.world_mut().resource_mut::<SelfClaim>().0 = Some(claim);
        None
    } else {
        Some(claim)
    };
    for value in [1, 2] {
        values.write(app.world_mut()).push(value);
        match step {
            BufferedStep::Scheduled => app.world_mut().run_schedule(Producer),
            #[cfg(feature = "imperative-signal-evaluation")]
            BufferedStep::Imperative => {
                let diffs = run_signal_subgraph_once(app.world_mut(), source.entity())
                    .and_then(downcast_signal_value::<Vec<VecDiff<i32>>>)
                    .expect("pending push");
                assert!(matches!(diffs.as_slice(), [VecDiff::Push { value: actual }] if *actual == value));
            }
        }
    }
    assert!(app.world().resource::<Outputs>().0.is_empty());
    match step {
        BufferedStep::Scheduled => {
            app.update();
        }
        #[cfg(feature = "imperative-signal-evaluation")]
        BufferedStep::Imperative => {
            let output = run_signal_subgraph_once(app.world_mut(), identity);
            if release {
                assert!(output.is_none(), "deactivated target must not return output");
            }
        }
    }
    assert_eq!(
        app.world().resource::<Outputs>().0,
        if release { vec![1] } else { vec![1, 2] }
    );
    if release {
        assert!(app.world().resource::<SelfClaim>().0.is_none());
        assert_eq!(app.world().get_entity(identity.entity()).is_ok(), retained);
        values.write(app.world_mut()).push(3);
        app.world_mut().run_schedule(Producer);
        app.update();
        assert_eq!(app.world().resource::<Outputs>().0, [1]);
    }
    if let Some(claim) = remaining_claim {
        claim.try_release(app.world_mut()).expect("control release");
    }
    source.try_release(app.world_mut()).expect("source release");
    drop(retained_terminal);
}

#[test]
fn self_release_on_first_buffered_event_must_stop_second_callback() {
    buffered_release_case(BufferedStep::Scheduled, true, true);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn imperative_self_release_on_first_buffered_event_must_stop_second_callback() {
    buffered_release_case(BufferedStep::Imperative, true, true);
}

#[test]
fn scheduled_self_release_can_reap_buffered_terminal() {
    buffered_release_case(BufferedStep::Scheduled, false, true);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn imperative_self_release_can_reap_buffered_terminal() {
    buffered_release_case(BufferedStep::Imperative, false, true);
}

#[test]
fn ordinary_buffered_batching_control() {
    buffered_release_case(BufferedStep::Scheduled, true, false);
    #[cfg(feature = "imperative-signal-evaluation")]
    buffered_release_case(BufferedStep::Imperative, true, false);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn imperative_evaluation_steps_established_and_newly_selected_dynamic_dependencies() {
    let mut app = app();
    let fresh = signal::once(signal::once(42i32).cache())
        .flatten()
        .map_in(|v: i32| v + 1)
        .register(app.world_mut());
    assert_eq!(
        run_signal_subgraph_once(app.world_mut(), fresh.entity()).and_then(downcast_signal_value::<i32>),
        Some(43),
        "evaluating a freshly registered dynamic signal must evaluate newly selected dependencies"
    );
    assert_eq!(
        run_signal_subgraph_once(app.world_mut(), fresh.entity()).and_then(downcast_signal_value::<i32>),
        None,
        "once-selected inner signal must not run twice"
    );
    fresh.release(app.world_mut());

    let vec = MutableVec::builder().values([10i32]).spawn(app.world_mut());
    let dynamic_vec = vec
        .signal_vec()
        .map_signal(|In(item): In<i32>| signal::once(item * 2).cache())
        .register(app.world_mut());
    let initial = run_signal_subgraph_once(app.world_mut(), dynamic_vec.entity())
        .and_then(downcast_signal_value::<Vec<VecDiff<i32>>>)
        .expect("initial dynamic vec evaluation");
    let mut observed = Vec::new();
    for diff in initial {
        diff.apply_to_vec(&mut observed);
    }
    assert_eq!(observed, [20]);

    vec.write(app.world_mut()).push(30);
    let next = run_signal_subgraph_once(app.world_mut(), dynamic_vec.entity())
        .and_then(downcast_signal_value::<Vec<VecDiff<i32>>>)
        .expect("established dynamic vec evaluation after source mutation");
    for diff in next {
        diff.apply_to_vec(&mut observed);
    }
    assert_eq!(observed, [20, 60]);
    dynamic_vec.release(app.world_mut());
}

#[test]
fn release_and_registration_preflight_do_not_clone_existing_cached_collections() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    let mut app = app();
    #[derive(Debug)]
    struct CloneProbe(Arc<AtomicUsize>);
    impl Clone for CloneProbe {
        fn clone(&self) -> Self {
            self.0.fetch_add(1, Ordering::Relaxed);
            Self(self.0.clone())
        }
    }
    let clones = Arc::new(AtomicUsize::new(0));
    let counted_vec = MutableVec::builder()
        .values((0..32).map(|_| CloneProbe(clones.clone())).collect::<Vec<_>>())
        .spawn(app.world_mut());
    let first = counted_vec
        .signal_vec()
        .for_each(|_: In<Vec<VecDiff<CloneProbe>>>| {})
        .register(app.world_mut());
    app.update();
    clones.store(0, Ordering::Relaxed);
    first.release(app.world_mut());
    assert_eq!(
        clones.load(Ordering::Relaxed),
        0,
        "releasing a consumer must not clone the cached collection just to validate readiness"
    );
}
