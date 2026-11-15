#![allow(missing_docs)]

extern crate alloc;

use alloc::sync::Arc;
use bevy_app::App;
use bevy_ecs::prelude::*;
use jonmo::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery, ReplayInitialization,
    SameSignalIdentity, ScalarSignalEntityRef, ScalarSignalMaterializer, ScalarSlotCommand, SignalRegistrationError,
    prelude::*,
};
use std::sync::Mutex;

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.update();
    app
}

fn pump(app: &mut App, frames: usize) {
    for _ in 0..frames {
        app.update();
    }
}

fn live_entities(world: &mut World) -> Vec<Entity> {
    world.query::<Entity>().iter(world).collect()
}

struct Source;

impl Signal for Source {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        ScalarSignalMaterializer::new(|registrar| {
            registrar.replayable_signal_source(41, |In(())| None::<i32>, OutputDelivery::broadcast())
        })
    }
}

struct Selections {
    signals: Vec<ReplayableBroadcast<i32>>,
    errors: Arc<Mutex<Vec<String>>>,
}

impl Signal for Selections {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        let Self { signals, errors } = *self;
        ScalarSignalMaterializer::new(move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(()), mut index: Local<usize>| {
                    let signal = signals.get(*index).cloned();
                    *index += 1;
                    signal.map(|signal| {
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
                        errors.lock().expect("error log lock").push(format!("{error:?}"));
                    }),
            )
        })
    }
}

struct Alias(ReplayableBroadcast<i32>);

impl Signal for Alias {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        ScalarSignalMaterializer::new(move |registrar| {
            registrar.materialize_signal(Box::new(self.0).into_signal_materializer())
        })
    }
}

fn probe_registration(signal: ReplayableBroadcast<i32>, app: &mut App) -> String {
    match signal.try_register(app.world_mut()) {
        Ok(claim) => {
            let entity = claim.entity();
            claim.try_release(app.world_mut()).expect("probe release");
            format!("Ok({entity:?})")
        }
        Err(error) => format!("Err({error:?})"),
    }
}

#[test]
fn all_keep_current_alias_must_be_reusable() {
    let mut app = app();
    let current = Source.cache().broadcast();
    let alias = Alias(current.clone()).cache().broadcast();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let owner = Selections {
        signals: vec![current.clone(), alias.clone(), alias.clone()],
        errors: errors.clone(),
    }
    .register(app.world_mut());

    app.update();
    assert!(probe_registration(current, &mut app).starts_with("Ok("));
    app.update();
    let first = probe_registration(alias.clone(), &mut app);
    pump(&mut app, 10);
    let later = probe_registration(alias, &mut app);
    let candidate_errors = errors.lock().expect("error log lock").clone();
    println!("alias: first = {first}; later = {later}; errors = {candidate_errors:?}");
    owner.try_release(app.world_mut()).expect("owner release");
    assert!(first.starts_with("Ok("), "alias should become Ready, observed {first}");
    assert!(later.starts_with("Ok("));
    assert!(candidate_errors.is_empty());
}

#[test]
fn all_keep_current_already_ready_identity_control() {
    let mut app = app();
    let current = Source.cache().broadcast();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let owner = Selections {
        signals: vec![current.clone(), current.clone(), current.clone()],
        errors: errors.clone(),
    }
    .register(app.world_mut());
    pump(&mut app, 12);
    let result = probe_registration(current, &mut app);
    println!(
        "ready identity control: registration = {result}; errors = {:?}",
        errors.lock().expect("error log lock")
    );
    assert!(result.starts_with("Ok("));
    assert!(errors.lock().expect("error log lock").is_empty());
    owner.try_release(app.world_mut()).expect("owner release");
}

struct UnusedPlanned(Arc<()>);

impl Signal for UnusedPlanned {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        ScalarSignalMaterializer::new(move |registrar| {
            let probe = self.0;
            let _unused: ScalarSignalEntityRef<'_, i32> = registrar.signal_entity(
                (),
                move |In(())| Some(Arc::strong_count(&probe) as i32),
                OutputDelivery::broadcast(),
            )?;
            registrar.signal_entity((), |In(())| Some(7), OutputDelivery::broadcast())
        })
    }
}

#[test]
fn returned_terminal_captured_resource_cleanup_control() {
    let mut app = app();
    let baseline = live_entities(app.world_mut()).len();
    let probe = Arc::new(());
    let captured_probe = probe.clone();
    let claim = ScalarSignalMaterializer::<i32>::new(move |registrar| {
        registrar.signal_entity(
            (),
            move |In(())| Some(Arc::strong_count(&captured_probe) as i32),
            OutputDelivery::broadcast(),
        )
    })
    .register(app.world_mut());
    claim.try_release(app.world_mut()).expect("control terminal release");
    pump(&mut app, 32);
    let remaining = live_entities(app.world_mut()).len();
    println!(
        "unused control: entities = {remaining}; probe strong count = {}",
        Arc::strong_count(&probe)
    );
    assert_eq!((remaining, Arc::strong_count(&probe)), (baseline, 1));
}

#[test]
fn unused_planned_entity_must_not_survive_terminal_cleanup() {
    let mut app = app();
    let baseline = live_entities(app.world_mut()).len();
    let probe = Arc::new(());
    let claim = UnusedPlanned(probe.clone()).register(app.world_mut());
    let terminal = claim.entity().entity();
    claim.try_release(app.world_mut()).expect("terminal release");
    pump(&mut app, 32);
    let remaining = live_entities(app.world_mut()).len();
    let retained_probe_count = Arc::strong_count(&probe);
    println!("unused: entities = {remaining}; probe strong count = {retained_probe_count}");
    assert!(
        app.world().get_entity(terminal).is_err(),
        "returned terminal must be cleaned up"
    );
    drop(app);
    assert_eq!(Arc::strong_count(&probe), 1);
    assert_eq!(
        (remaining, retained_probe_count),
        (baseline, 1),
        "unused planned entity and captured resource leaked after all terminal ownership was released"
    );
}

#[test]
fn unused_raw_chains_are_cleaned_before_registration_returns() {
    for replayable in [false, true] {
        let mut app = app();
        let probe = Arc::new(());
        let captured_probe = probe.clone();
        let claim = ScalarSignalMaterializer::<i32>::new(move |registrar| {
            let system = move |In(())| Some(Arc::strong_count(&captured_probe) as i32);
            let mut unused = if replayable {
                registrar.replayable_signal_source(0, system, OutputDelivery::broadcast())?
            } else {
                registrar.signal_entity((), system, OutputDelivery::broadcast())?
            };
            for _ in 0..4 {
                unused = if replayable {
                    registrar.replayable_signal_entity(
                        unused,
                        |In(value): In<i32>| Some(value),
                        ReplayInitialization::Warming,
                        OutputDelivery::broadcast(),
                    )?
                } else {
                    registrar.signal_entity(unused, |In(value): In<i32>| Some(value), OutputDelivery::broadcast())?
                };
            }
            registrar.signal_entity((), |In(())| Some(7), OutputDelivery::broadcast())
        })
        .register(app.world_mut());
        assert_eq!(live_entities(app.world_mut()), [claim.entity().entity()]);
        assert_eq!(Arc::strong_count(&probe), 1);
        claim.try_release(app.world_mut()).expect("terminal release");
        assert!(live_entities(app.world_mut()).is_empty());
    }
}

#[test]
fn unused_branch_cleanup_preserves_the_shared_dependency() {
    let mut app = app();
    let probe = Arc::new(());
    let captured_probe = probe.clone();
    let values = Arc::new(Mutex::new(Vec::new()));
    let observed = values.clone();
    let claim = ScalarSignalMaterializer::<i32>::new(move |registrar| {
        let source = registrar.replayable_signal_source(41, |In(())| None::<i32>, OutputDelivery::broadcast())?;
        let _unused: ScalarSignalEntityRef<'_, i32> = registrar.signal_entity(
            source,
            move |In(value): In<i32>| Some(value + Arc::strong_count(&captured_probe) as i32),
            OutputDelivery::broadcast(),
        )?;
        registrar.signal_entity(
            source,
            move |In(value): In<i32>| {
                observed.lock().expect("observations lock").push(value);
                Some(value)
            },
            OutputDelivery::broadcast(),
        )
    })
    .register(app.world_mut());
    assert_eq!(live_entities(app.world_mut()).len(), 2);
    assert_eq!(Arc::strong_count(&probe), 1);
    pump(&mut app, 4);
    assert_eq!(*values.lock().expect("observations lock"), [41]);
    claim.try_release(app.world_mut()).expect("terminal release");
    assert!(live_entities(app.world_mut()).is_empty());
}

#[derive(Clone)]
enum RetainedChild {
    Ordinary(Broadcast<i32>),
    Replayable(ReplayableBroadcast<i32>),
}

impl RetainedChild {
    fn into_materializer(self) -> ScalarSignalMaterializer<i32> {
        match self {
            Self::Ordinary(signal) => Box::new(signal).into_signal_materializer(),
            Self::Replayable(signal) => Box::new(signal).into_signal_materializer(),
        }
    }

    fn try_register(self, world: &mut World) -> Result<jonmo::graph::SignalClaim, SignalRegistrationError> {
        match self {
            Self::Ordinary(signal) => signal.try_register(world),
            Self::Replayable(signal) => signal.try_register(world),
        }
    }
}

fn retained_independent_child(replayable: bool) {
    let mut app = app();
    let probe = Arc::new(());
    let captured_probe = probe.clone();
    let runs = Arc::new(core::sync::atomic::AtomicUsize::new(0));
    let source_runs = runs.clone();
    let child = jonmo::signal::from_system::<i32, _, _, _>(move |In(())| {
        source_runs.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        Some(Arc::strong_count(&captured_probe) as i32)
    });
    let child = if replayable {
        RetainedChild::Replayable(child.cache().broadcast())
    } else {
        RetainedChild::Ordinary(child.broadcast())
    };
    let side_child = child.clone();
    let outer = ScalarSignalMaterializer::<i32>::new(move |registrar| {
        let _independent = registrar.materialize_signal(side_child.into_materializer())?;
        registrar.signal_entity((), |In(())| Some(7), OutputDelivery::broadcast())
    })
    .register(app.world_mut());
    outer.try_release(app.world_mut()).expect("outer release");
    pump(&mut app, 4);
    assert_eq!(Arc::strong_count(&probe), 2);
    assert_eq!(runs.load(core::sync::atomic::Ordering::SeqCst) > 0, replayable);
    let claim = child
        .clone()
        .try_register(app.world_mut())
        .expect("independent child remains reusable");
    pump(&mut app, 2);
    assert!(runs.load(core::sync::atomic::Ordering::SeqCst) > 0);
    claim.try_release(app.world_mut()).expect("child release");
    assert!(!live_entities(app.world_mut()).is_empty());
    drop(child);
    pump(&mut app, 4);
    assert!(live_entities(app.world_mut()).is_empty());
    assert_eq!(Arc::strong_count(&probe), 1);
}

#[test]
fn unused_independent_child_retainer_preserves_dormant_materialization() {
    retained_independent_child(false);
}

#[test]
fn unused_independent_replay_child_stays_hot_until_final_retainer_drop() {
    retained_independent_child(true);
}

struct AliasWithUnused {
    selected: ReplayableBroadcast<i32>,
    probe: Arc<()>,
}

impl Signal for AliasWithUnused {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        ScalarSignalMaterializer::new(move |registrar| {
            let probe = self.probe;
            let unused = registrar.replayable_signal_source(
                0,
                move |In(())| Some(Arc::strong_count(&probe) as i32),
                OutputDelivery::broadcast(),
            )?;
            let _unused: ScalarSignalEntityRef<'_, i32> = registrar.replayable_signal_entity(
                unused,
                |In(value): In<i32>| Some(value),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )?;
            registrar.materialize_signal(Box::new(self.selected).into_signal_materializer())
        })
    }
}

#[test]
fn dynamic_candidate_cleanup_preserves_private_adapter_claims_and_keep_current() {
    for keep_current in [false, true] {
        let mut app = app();
        let probe = Arc::new(());
        let selected = Source.cache().broadcast();
        let candidate = AliasWithUnused {
            selected: selected.clone(),
            probe: probe.clone(),
        }
        .cache()
        .broadcast();
        let errors = Arc::new(Mutex::new(Vec::new()));
        let signals = if keep_current {
            vec![selected.clone(), candidate.clone(), candidate.clone()]
        } else {
            vec![candidate.clone()]
        };
        let values = Arc::new(Mutex::new(Vec::new()));
        let observed = values.clone();
        let owner = Selections {
            signals,
            errors: errors.clone(),
        }
        .map(move |In(value): In<i32>| {
            observed.lock().expect("observations lock").push(value);
        })
        .register(app.world_mut());
        pump(&mut app, 8);
        assert_eq!(Arc::strong_count(&probe), 1);
        assert!(probe_registration(candidate.clone(), &mut app).starts_with("Ok("));
        assert!(errors.lock().expect("error log lock").is_empty());
        assert_eq!(*values.lock().expect("observations lock"), [41]);
        owner.try_release(app.world_mut()).expect("owner release");
        drop(candidate);
        drop(selected);
        pump(&mut app, 4);
        assert!(live_entities(app.world_mut()).is_empty());
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, bevy_ecs::schedule::ScheduleLabel)]
struct UnregisteredSchedule;

struct RejectedAlias(ReplayableBroadcast<i32>, Arc<()>);

impl Signal for RejectedAlias {
    type Item = i32;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        ScalarSignalMaterializer::new(move |registrar| {
            let probe = self.1;
            let _unused: ScalarSignalEntityRef<'_, i32> = registrar.signal_entity(
                (),
                move |In(())| Some(Arc::strong_count(&probe) as i32),
                OutputDelivery::broadcast(),
            )?;
            let selected = registrar.materialize_signal(Box::new(self.0).into_signal_materializer())?;
            registrar.apply_schedule(selected, UnregisteredSchedule)
        })
    }
}

#[test]
fn rejected_keep_current_alias_discards_unpublished_side_entities() {
    let mut app = app();
    let current = Source.cache().broadcast();
    let probe = Arc::new(());
    let alias = RejectedAlias(current.clone(), probe.clone()).cache().broadcast();
    let errors = Arc::new(Mutex::new(Vec::new()));
    let owner = Selections {
        signals: vec![current.clone(), alias.clone(), alias.clone()],
        errors: errors.clone(),
    }
    .register(app.world_mut());
    app.update();
    let mut before = live_entities(app.world_mut());
    before.sort();
    pump(&mut app, 4);
    let mut after = live_entities(app.world_mut());
    after.sort();
    assert_eq!(after, before);
    assert_eq!(Arc::strong_count(&probe), 1);
    assert_eq!(
        *errors.lock().expect("error log lock"),
        ["ScheduleNotRegistered", "SharedMaterializerRejected"]
    );
    assert_eq!(probe_registration(alias, &mut app), "Err(SharedMaterializerRejected)");
    assert!(probe_registration(current, &mut app).starts_with("Ok("));
    owner.try_release(app.world_mut()).expect("owner release");
    pump(&mut app, 4);
    assert!(live_entities(app.world_mut()).is_empty());
}
