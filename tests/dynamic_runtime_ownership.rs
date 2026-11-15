//! Reentrant lifecycle operations must not require borrowing a dynamic runtime outside its owner.

use bevy_app::App;
use bevy_ecs::{prelude::*, system::SystemIdMarker};
use jonmo::{OutputDelivery, ScalarSignalMaterializer, SignalRegistrationError, prelude::*};

#[derive(Clone)]
enum RetainedSignal {
    Ordinary(Broadcast<i32>),
    Replayable(ReplayableBroadcast<i32>),
}

impl RetainedSignal {
    fn try_register(self, world: &mut World) -> Result<SignalClaim, SignalRegistrationError> {
        match self {
            Self::Ordinary(signal) => signal.try_register(world),
            Self::Replayable(signal) => signal.try_register(world),
        }
    }
}

#[derive(Resource)]
struct ReentrantOwner {
    claim: Option<SignalClaim>,
    retained: RetainedSignal,
    ran: bool,
}

#[derive(Resource, Default)]
struct Values(Vec<i32>);

fn reregister(world: &mut World) {
    let (claim, retained) = {
        let mut state = world.resource_mut::<ReentrantOwner>();
        state.ran = true;
        (state.claim.take().expect("owned claim"), state.retained.clone())
    };
    let identity = claim.entity();
    claim.try_release(world).expect("reentrant release");
    let claim = retained.try_register(world).expect("reentrant registration");
    assert_eq!(claim.entity(), identity);
    world.resource_mut::<ReentrantOwner>().claim = Some(claim);
}

struct ReregisterOnInitialize(bool);

impl FromWorld for ReregisterOnInitialize {
    fn from_world(world: &mut World) -> Self {
        reregister(world);
        Self(false)
    }
}

#[test]
fn candidate_spawn_hook_can_release_and_reregister_retained_owner() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Values>();
    app.world_mut()
        .register_component_hooks::<SystemIdMarker>()
        .on_add(|mut world, _| {
            if world.get_resource::<ReentrantOwner>().is_some_and(|state| !state.ran) {
                world.resource_mut::<ReentrantOwner>().ran = true;
                world.commands().queue(reregister);
            }
        });
    let selected = signal::once(7).map(|In(value): In<i32>| value).cache();
    let retained = signal::once(selected).flatten().broadcast();
    let claim = retained.clone().register(app.world_mut());
    app.insert_resource(ReentrantOwner {
        claim: Some(claim),
        retained: RetainedSignal::Replayable(retained.clone()),
        ran: false,
    });
    app.update();
    assert!(app.world().resource::<ReentrantOwner>().ran);
    let observer = retained
        .map(|In(value): In<i32>, mut values: ResMut<Values>| values.0.push(value))
        .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<Values>().0, [7]);
    observer.release(app.world_mut());
    let state = app
        .world_mut()
        .remove_resource::<ReentrantOwner>()
        .expect("owner state");
    state.claim.expect("reregistered claim").release(app.world_mut());
    drop(state.retained);
    app.update();
}

#[test]
fn final_ancestor_release_and_reregistration_cancels_old_dynamic_materialization() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Values>();
    app.world_mut()
        .register_component_hooks::<SystemIdMarker>()
        .on_add(|mut world, _| {
            if world.get_resource::<ReentrantOwner>().is_some_and(|state| !state.ran) {
                world.resource_mut::<ReentrantOwner>().ran = true;
                world.commands().queue(reregister);
            }
        });
    let retained =
        signal::from_system::<BoxedReplayableSignal<i32>, _, _, _>(|In(())| signal::once(7).cache().boxed_replayable())
            .flatten()
            .map(|In(value): In<i32>, mut values: ResMut<Values>| {
                values.0.push(value);
                value
            })
            .broadcast();
    let claim = retained.clone().register(app.world_mut());
    app.insert_resource(ReentrantOwner {
        claim: Some(claim),
        retained: RetainedSignal::Ordinary(retained),
        ran: false,
    });
    app.update();
    assert!(app.world().resource::<ReentrantOwner>().ran);
    assert!(
        app.world().resource::<Values>().0.is_empty(),
        "retired in-flight commands do not survive reactivation"
    );
    app.update();
    assert_eq!(
        app.world().resource::<Values>().0,
        [7],
        "new manager updates still execute after reactivation"
    );
    let state = app
        .world_mut()
        .remove_resource::<ReentrantOwner>()
        .expect("owner state");
    state.claim.expect("reregistered claim").release(app.world_mut());
    drop(state.retained);
    app.update();
}

#[test]
fn selected_local_can_release_and_reregister_retained_owner() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Values>();
    let selected = signal::from_system::<i32, _, _, _>(|In(()), mut local: Local<ReregisterOnInitialize>| {
        (!core::mem::replace(&mut local.0, true)).then_some(9)
    })
    .cache();
    let retained = signal::once(selected).flatten().broadcast();
    let claim = retained.clone().register(app.world_mut());
    app.insert_resource(ReentrantOwner {
        claim: Some(claim),
        retained: RetainedSignal::Replayable(retained.clone()),
        ran: false,
    });
    app.update();
    assert!(app.world().resource::<ReentrantOwner>().ran);
    let observer = retained
        .map(|In(value): In<i32>, mut values: ResMut<Values>| values.0.push(value))
        .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<Values>().0, [9]);
    observer.release(app.world_mut());
    let state = app
        .world_mut()
        .remove_resource::<ReentrantOwner>()
        .expect("owner state");
    state.claim.expect("reregistered claim").release(app.world_mut());
    drop(state.retained);
    app.update();
}

struct CustomSelected(ScalarSignalMaterializer<i32>);

impl Signal for CustomSelected {
    type Item = i32;
    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
        self.0
    }
}

#[test]
fn candidate_cleanup_hook_can_release_owner_without_losing_committed_subscriptions() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Allocations>();
    app.world_mut()
        .register_component_hooks::<SystemIdMarker>()
        .on_add(|mut world, context| {
            if world.contains_resource::<OwnerClaim>() {
                world.resource_mut::<Allocations>().0.push(context.entity);
            }
        })
        .on_remove(|mut world, _| {
            if let Some(mut state) = world.get_resource_mut::<OwnerClaim>()
                && let Some(claim) = state.0.take()
            {
                world.commands().queue(move |world: &mut World| claim.release(world));
            }
        });
    let selected = CustomSelected(ScalarSignalMaterializer::new(|registrar| {
        registrar.signal_entity((), |In(())| 99_i32, OutputDelivery::broadcast())?;
        registrar.replayable_signal_source(7, |In(())| None::<i32>, OutputDelivery::broadcast())
    }))
    .cache()
    .broadcast();
    let retained = signal::always(selected)
        .flatten()
        .map(|In(value): In<i32>| value)
        .broadcast();
    let claim = retained.clone().register(app.world_mut());
    app.insert_resource(OwnerClaim(Some(claim)));
    let mut schedule = app
        .world_mut()
        .resource_mut::<Schedules>()
        .remove(bevy_app::Last)
        .expect("Jonmo schedule");
    let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| schedule.run(app.world_mut())));
    assert!(app.world().resource::<OwnerClaim>().0.is_none());
    let [unused, selected, adapter] = app.world().resource::<Allocations>().0.as_slice() else {
        panic!("expected unused system, selected source, and adapter");
    };
    assert!(app.world().get_entity(*unused).is_err());
    assert!(
        app.world().get_entity(*adapter).is_err(),
        "unadopted adapter claim survived retirement"
    );
    assert!(
        app.world().get_entity(*selected).is_ok(),
        "manager still retains the selected source"
    );
    drop(retained);
    schedule.run(app.world_mut());
    // Reaping the manager drops its captured selected chain after this drain's inbox boundary.
    schedule.run(app.world_mut());
    let survivors = app
        .world()
        .resource::<Allocations>()
        .0
        .iter()
        .copied()
        .filter(|entity| app.world().get::<SystemIdMarker>(*entity).is_some())
        .collect::<Vec<_>>();

    assert!(
        result.is_ok(),
        "owner retirement during committed candidate cleanup panicked"
    );
    assert!(survivors.is_empty(), "committed candidate systems were leaked");
}

#[derive(Resource)]
struct OwnerClaim(Option<SignalClaim>);

#[derive(Resource, Default)]
struct Allocations(Vec<Entity>);

#[test]
fn cancelling_owner_does_not_reject_independent_selected_alias() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Allocations>();
    app.init_resource::<Values>();
    app.world_mut()
        .register_component_hooks::<SystemIdMarker>()
        .on_add(|mut world, context| {
            if world.contains_resource::<OwnerClaim>() {
                world.resource_mut::<Allocations>().0.push(context.entity);
                if let Some(claim) = world.resource_mut::<OwnerClaim>().0.take() {
                    world.commands().queue(move |world: &mut World| claim.release(world));
                }
            }
        });
    let selected = signal::once(7).map(|In(value): In<i32>| value).cache().broadcast();
    let retained = signal::once(selected.clone())
        .flatten()
        .map(|In(value): In<i32>| value)
        .broadcast();
    let claim = retained.clone().register(app.world_mut());
    app.insert_resource(OwnerClaim(Some(claim)));
    app.update();
    assert!(app.world().resource::<OwnerClaim>().0.is_none());
    let allocations = app.world().resource::<Allocations>().0.clone();
    assert!(!allocations.is_empty());
    // Cancellation must publish the valid selected plan for its independently retained alias.
    let result = selected.clone().try_register(app.world_mut());
    assert!(
        result.is_ok(),
        "independent selected alias was poisoned by owner cancellation: {result:?}"
    );
    let claim = result.expect("valid selected alias");
    assert!(allocations.contains(&*claim.entity()));
    assert_eq!(
        app.world().resource::<Allocations>().0,
        allocations,
        "alias was rematerialized"
    );
    let observer = selected
        .map(|In(value): In<i32>, mut values: ResMut<Values>| values.0.push(value))
        .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<Values>().0, [7]);
    observer.release(app.world_mut());
    claim.release(app.world_mut());
    drop(retained);
    app.update();
    app.update();
    assert!(
        allocations
            .iter()
            .all(|entity| app.world().get_entity(*entity).is_err()),
        "cancelled materialization leaked entities after all retained aliases were dropped"
    );
}

#[derive(Resource, Default)]
struct Selection(Option<ReplayableBroadcast<i32>>);

#[test]
fn retirement_preserves_maintenance_when_new_alias_reuses_existing_selected_terminal() {
    existing_alias_retirement(false);
}

#[test]
fn retirement_does_not_duplicate_maintenance_when_old_selected_alias_is_retained() {
    existing_alias_retirement(true);
}

fn existing_alias_retirement(keep_existing_alias: bool) {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Values>();
    app.init_resource::<Allocations>();
    app.world_mut()
        .register_component_hooks::<SystemIdMarker>()
        .on_add(|mut world, context| {
            world.resource_mut::<Allocations>().0.push(context.entity);
            if let Some(mut owner) = world.get_resource_mut::<OwnerClaim>()
                && let Some(claim) = owner.0.take()
            {
                world.commands().queue(move |world: &mut World| claim.release(world));
            }
        });
    let selected = signal::once(7).cache().broadcast();
    let existing_alias = keep_existing_alias.then(|| selected.clone());
    let identity_claim = selected.clone().register(app.world_mut());
    let identity = identity_claim.entity();
    identity_claim.release(app.world_mut());
    app.insert_resource(Selection(Some(selected.clone())));
    let claim = signal::from_system::<ReplayableBroadcast<i32>, _, _, _>(|In(()), mut selection: ResMut<Selection>| {
        selection.0.take()
    })
    .flatten()
    .map(|In(value): In<i32>, mut values: ResMut<Values>| values.0.push(value))
    .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<Values>().0, [7]);
    let alias = CustomSelected(ScalarSignalMaterializer::new(move |registrar| {
        let selected = registrar.materialize_signal(Box::new(selected).into_signal_materializer())?;
        // KeepCurrent needs no adapter, but this allocation still lets a hook retire the owner
        // after the old materializer's last live chain has been consumed into the new alias.
        registrar.signal_entity((), |In(())| 99_i32, OutputDelivery::broadcast())?;
        Ok(selected)
    }))
    .cache()
    .broadcast();
    app.world_mut().resource_mut::<Selection>().0 = Some(alias.clone());
    app.insert_resource(OwnerClaim(Some(claim)));
    app.update();
    assert!(app.world().resource::<OwnerClaim>().0.is_none());
    assert_eq!(app.world().resource::<Values>().0, [7]);
    let allocations = app.world().resource::<Allocations>().0.clone();
    let claim = alias
        .clone()
        .try_register(app.world_mut())
        .expect("alias retains valid replay maintenance");
    assert_eq!(claim.entity(), identity);
    claim.release(app.world_mut());
    assert_eq!(
        app.world().resource::<Allocations>().0,
        allocations,
        "alias was rematerialized"
    );
    let observer = alias
        .clone()
        .map(|In(value): In<i32>, mut values: ResMut<Values>| values.0.push(value))
        .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<Values>().0, [7, 7]);
    observer.release(app.world_mut());
    drop(alias);
    drop(existing_alias);
    app.update();
    app.update();
    assert!(
        app.world_mut()
            .query_filtered::<Entity, With<SystemIdMarker>>()
            .iter(app.world())
            .next()
            .is_none(),
        "final alias drop must balance exactly one maintenance claim and reap every system"
    );
}

#[test]
fn allocation_retirement_keeps_validated_existing_dependencies_until_commit_finishes() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<Allocations>();
    app.init_resource::<Values>();
    app.world_mut()
        .register_component_hooks::<SystemIdMarker>()
        .on_add(|mut world, context| {
            world.resource_mut::<Allocations>().0.push(context.entity);
            if let Some(mut owner) = world.get_resource_mut::<OwnerClaim>()
                && let Some(claim) = owner.0.take()
            {
                world.commands().queue(move |world: &mut World| claim.release(world));
            }
        });
    let selected = signal::once(7).cache().broadcast();
    app.insert_resource(Selection(Some(selected.clone())));
    let claim = signal::from_system::<ReplayableBroadcast<i32>, _, _, _>(|In(()), mut selection: ResMut<Selection>| {
        selection.0.take()
    })
    .flatten()
    .map(|In(value): In<i32>, mut values: ResMut<Values>| values.0.push(value))
    .register(app.world_mut());
    app.update();
    assert_eq!(app.world().resource::<Values>().0, [7]);
    // This last selected chain becomes a dependency of a new candidate during planning. The
    // allocation hook then retires the old owner and its current adapter before commit finishes.
    let replacement = CustomSelected(ScalarSignalMaterializer::new(move |registrar| {
        let selected = registrar.materialize_signal(Box::new(selected).into_signal_materializer())?;
        registrar.signal_entity(selected, |In(value): In<i32>| value, OutputDelivery::broadcast())
    }))
    .cache()
    .broadcast();
    app.world_mut().resource_mut::<Selection>().0 = Some(replacement);
    app.insert_resource(OwnerClaim(Some(claim)));
    app.update();
    assert!(app.world().resource::<OwnerClaim>().0.is_none());
    assert_eq!(app.world().resource::<Values>().0, [7], "retired output was forwarded");
    app.update();
    assert!(
        app.world()
            .resource::<Allocations>()
            .0
            .iter()
            .all(|entity| app.world().get_entity(*entity).is_err()),
        "existing selected dependency or cancelled candidate survived retirement"
    );
}
