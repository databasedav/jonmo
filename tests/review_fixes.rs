//! End-to-end regressions for the Move-First post-checkpoint review fixes:
//! - `jonmo::prelude` ergonomics and `jonmo::registration` / `jonmo::dynamic` module boundaries
//! - `BoxedReplayable*` inherent `.cache()`, `.broadcast()`, `.schedule()`, `.boxed_replayable()`
//! - Dynamic `.schedule::<S>()` propagation to runtime adapters and untagged inner chains
//! - Mid-pass dynamic level promotion single-pass topological execution
//! - `ScalarDynamicRuntime` `KeepCurrent` across `ScalarSlotCommand::Clear`
//! - Unretained `1 -> 0 -> 1` reactivation for stateful dynamic combinators
//! - `SignalVecExt::flatten` and multi-`Builder::spawn_on_entity` block isolation

#![allow(missing_docs)]

extern crate alloc;

use alloc::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    vec,
    vec::Vec,
};
use bevy_app::{App, Last};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel, system::SystemIdMarker};
use bevy_platform::sync::Mutex;
#[cfg(feature = "builder")]
use jonmo::Builder;
use jonmo::{
    dynamic::{
        DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, SameSignalIdentity, ScalarSlotCommand,
    },
    prelude::*,
    registration::{OutputDelivery, ReplayInitialization, ScalarSignalMaterializer},
};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
struct CustomSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
struct ExplicitInnerSchedule;

fn system_entity_count(world: &mut World) -> usize {
    world
        .query_filtered::<Entity, With<SystemIdMarker>>()
        .iter(world)
        .count()
}

#[test]
fn prelude_exposes_everyday_types_and_keeps_authoring_types_namespaced() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<CustomSchedule>());

    let vec_state = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let map_state = MutableBTreeMap::builder().values([(1, 10)]).spawn(app.world_mut());

    let scheduled_scalar: Scheduled<CustomSchedule, i32> = signal::always(5).schedule::<CustomSchedule>();
    let scheduled_vec: ScheduledVec<CustomSchedule, i32> =
        vec_state.signal_vec().map_in(|x| x + 1).schedule::<CustomSchedule>();
    let scheduled_map: ScheduledMap<CustomSchedule, i32, i32> = map_state
        .signal_map()
        .map_value(|In(v): In<i32>| v + 1)
        .schedule::<CustomSchedule>();

    let vec_state_out: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
    let map_state_out: Arc<Mutex<BTreeMap<i32, i32>>> = Arc::new(Mutex::new(BTreeMap::new()));

    let claim_scalar: SignalClaim = scheduled_scalar.register(app.world_mut());
    let claim_vec: SignalClaim = scheduled_vec
        .for_each({
            let vec_state_out = Arc::clone(&vec_state_out);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                let mut out = vec_state_out.lock().unwrap_or_else(|e| e.into_inner());
                for diff in diffs {
                    diff.apply_to_vec(&mut out);
                }
            }
        })
        .schedule::<CustomSchedule>()
        .register(app.world_mut());
    let claim_map: SignalClaim = scheduled_map
        .for_each({
            let map_state_out = Arc::clone(&map_state_out);
            move |In(diffs): In<Vec<MapDiff<i32, i32>>>| {
                let mut out = map_state_out.lock().unwrap_or_else(|e| e.into_inner());
                for diff in diffs {
                    if let MapDiff::Replace { entries } = diff {
                        *out = entries.into_iter().collect();
                    }
                }
            }
        })
        .schedule::<CustomSchedule>()
        .register(app.world_mut());

    app.world_mut().run_schedule(CustomSchedule);
    app.update();

    assert_eq!(*vec_state_out.lock().unwrap_or_else(|e| e.into_inner()), vec![2, 3]);
    assert_eq!(
        *map_state_out.lock().unwrap_or_else(|e| e.into_inner()),
        BTreeMap::from([(1, 11)])
    );

    let owner = app
        .world_mut()
        .spawn(SignalClaims::from([claim_scalar, claim_vec, claim_map]))
        .id();
    app.world_mut().despawn(owner);
}

#[test]
fn boxed_replayable_wrappers_resolve_replayable_inherent_methods_with_prelude_in_scope() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<CustomSchedule>());

    // Scalar BoxedReplayableSignal: .cache() is zero-cost idempotent, .broadcast() ->
    // ReplayableBroadcast, .schedule() -> ReplayableScheduled.
    let before = system_entity_count(app.world_mut());
    let boxed_scalar: BoxedReplayableSignal<i32> = signal::always(42).cache().boxed_replayable();
    let cached_again: BoxedReplayableSignal<i32> = boxed_scalar.cache().boxed_replayable();
    let scheduled_scalar: ReplayableScheduled<CustomSchedule, i32> = cached_again.schedule::<CustomSchedule>();
    let broadcast_scalar: ReplayableBroadcast<i32> = scheduled_scalar.broadcast();

    let scalar_seen = Arc::new(Mutex::new(Vec::new()));
    let scalar_claim = signal::always(true)
        .switch({
            let broadcast_scalar = broadcast_scalar.clone();
            move |In(_): In<bool>| broadcast_scalar.clone()
        })
        .map({
            let scalar_seen = Arc::clone(&scalar_seen);
            move |In(v): In<i32>| {
                scalar_seen.lock().unwrap_or_else(|e| e.into_inner()).push(v);
            }
        })
        .register(app.world_mut());
    let after_scalar = system_entity_count(app.world_mut());
    assert!(after_scalar > before);

    // Vec BoxedReplayableSignalVec
    let vec_source = MutableVec::builder().values([10, 20]).spawn(app.world_mut());
    let boxed_vec: BoxedReplayableSignalVec<i32> = vec_source.signal_vec().boxed_replayable();
    let broadcast_vec: ReplayableBroadcastVec<i32> = boxed_vec.cache().schedule::<CustomSchedule>().broadcast();
    let vec_seen = Arc::new(Mutex::new(Vec::new()));
    let vec_claim = signal::always(true)
        .switch_signal_vec({
            let broadcast_vec = broadcast_vec.clone();
            move |In(_): In<bool>| broadcast_vec.clone()
        })
        .for_each({
            let vec_seen = Arc::clone(&vec_seen);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                for diff in diffs {
                    diff.apply_to_vec(&mut vec_seen.lock().unwrap_or_else(|e| e.into_inner()));
                }
            }
        })
        .register(app.world_mut());

    // Map BoxedReplayableSignalMap
    let map_source = MutableBTreeMap::builder().values([(7, 70)]).spawn(app.world_mut());
    let boxed_map: BoxedReplayableSignalMap<i32, i32> = map_source.signal_map().boxed_replayable();
    let broadcast_map: ReplayableBroadcastMap<i32, i32> = boxed_map.cache().schedule::<CustomSchedule>().broadcast();
    let map_seen = Arc::new(Mutex::new(BTreeMap::new()));
    let map_claim = signal::always(true)
        .switch_signal_map({
            let broadcast_map = broadcast_map.clone();
            move |In(_): In<bool>| broadcast_map.clone()
        })
        .for_each({
            let map_seen = Arc::clone(&map_seen);
            move |In(diffs): In<Vec<MapDiff<i32, i32>>>| {
                let mut seen = map_seen.lock().unwrap_or_else(|e| e.into_inner());
                for diff in diffs {
                    if let MapDiff::Replace { entries } = diff {
                        *seen = entries.into_iter().collect();
                    }
                }
            }
        })
        .register(app.world_mut());

    // 1st `app.update()` runs the `PostUpdate` `switch` selectors, subscribing to the scheduled
    // inner branches; `run_schedule(CustomSchedule)` warms `signal::always(42)` on
    // `CustomSchedule`; 2nd `app.update()` delivers the warmed scalar output on `PostUpdate`.
    app.update();
    app.world_mut().run_schedule(CustomSchedule);
    app.update();

    assert_eq!(*scalar_seen.lock().unwrap_or_else(|e| e.into_inner()), vec![42]);
    assert_eq!(*vec_seen.lock().unwrap_or_else(|e| e.into_inner()), vec![10, 20]);
    assert_eq!(
        *map_seen.lock().unwrap_or_else(|e| e.into_inner()),
        BTreeMap::from([(7, 70)])
    );

    scalar_claim.release(app.world_mut());
    vec_claim.release(app.world_mut());
    map_claim.release(app.world_mut());
}

#[test]
fn dynamic_schedule_propagates_to_adapter_and_untagged_inner_without_overriding_explicit_inner_schedule() {
    #[derive(Resource, Default)]
    struct SelectorState(Option<bool>);

    #[derive(Resource, Default)]
    struct UntaggedSource(Option<i32>);

    #[derive(Resource, Default)]
    struct ExplicitSource(Option<i32>);

    let mut app = App::new();
    app.add_plugins(
        JonmoPlugin::new::<Last>()
            .with_schedule::<CustomSchedule>()
            .with_schedule::<ExplicitInnerSchedule>(),
    );
    app.insert_resource(SelectorState(Some(false)));
    app.insert_resource(UntaggedSource(Some(100)));
    app.insert_resource(ExplicitSource(None));

    let selector = ScalarSignalMaterializer::new(|registrar| {
        registrar.signal_entity(
            (),
            |In(()), mut state: ResMut<SelectorState>| state.0.take(),
            OutputDelivery::move_only(),
        )
    });

    let make_untagged_source = || {
        ScalarSignalMaterializer::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(()), mut src: ResMut<UntaggedSource>| src.0.take(),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        })
    };

    let make_explicit_source = || {
        ScalarSignalMaterializer::new(|registrar| {
            registrar.replayable_signal_entity(
                (),
                |In(()), mut src: ResMut<ExplicitSource>| src.0.take(),
                ReplayInitialization::Warming,
                OutputDelivery::broadcast(),
            )
        })
    };

    struct RawSignal<T: 'static>(ScalarSignalMaterializer<T>);
    impl<T: 'static> Signal for RawSignal<T> {
        type Item = T;
        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<T> {
            self.0
        }
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let claim = RawSignal(selector)
        .switch(move |In(use_explicit): In<bool>| {
            if use_explicit {
                RawSignal(make_explicit_source())
                    .map(|In(v): In<i32>| v * 10)
                    .cache()
                    .schedule::<ExplicitInnerSchedule>()
                    .boxed_replayable()
            } else {
                // Untagged inner chain: must inherit CustomSchedule from the outer
                // .schedule::<CustomSchedule>()!
                RawSignal(make_untagged_source())
                    .map(|In(v): In<i32>| v + 1)
                    .cache()
                    .boxed_replayable()
            }
        })
        .schedule::<CustomSchedule>()
        .map({
            let seen = Arc::clone(&seen);
            move |In(v): In<i32>| {
                seen.lock().unwrap_or_else(|e| e.into_inner()).push(v);
            }
        })
        .schedule::<CustomSchedule>()
        .register(app.world_mut());

    // Running ONLY CustomSchedule (never Last!) must materialize and run both the untagged inner
    // chain and adapter.
    app.world_mut().run_schedule(CustomSchedule);
    assert_eq!(*seen.lock().unwrap_or_else(|e| e.into_inner()), vec![101]);

    // Subsequent update on the untagged inner chain also runs purely inside CustomSchedule.
    app.world_mut().resource_mut::<UntaggedSource>().0 = Some(200);
    app.world_mut().run_schedule(CustomSchedule);
    assert_eq!(*seen.lock().unwrap_or_else(|e| e.into_inner()), vec![101, 201]);

    // Switching to the explicitly scheduled inner chain must NOT run that inner chain until
    // ExplicitInnerSchedule runs.
    app.world_mut().resource_mut::<SelectorState>().0 = Some(true);
    app.world_mut().resource_mut::<ExplicitSource>().0 = Some(30);
    app.world_mut().run_schedule(CustomSchedule);
    assert_eq!(
        *seen.lock().unwrap_or_else(|e| e.into_inner()),
        vec![101, 201],
        "explicitly scheduled inner chain must wait for ExplicitInnerSchedule"
    );

    app.world_mut().run_schedule(ExplicitInnerSchedule);
    app.world_mut().run_schedule(CustomSchedule);
    assert_eq!(*seen.lock().unwrap_or_else(|e| e.into_inner()), vec![101, 201, 300]);

    claim.release(app.world_mut());
}

#[test]
fn mid_pass_dynamic_level_promotion_executes_once_without_stale_glitch() {
    #[derive(Resource, Default)]
    struct TickValue(Option<i32>);

    struct TickSource;
    impl Signal for TickSource {
        type Item = i32;
        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
            ScalarSignalMaterializer::new(|registrar| {
                registrar.replayable_signal_source(
                    1,
                    |In(()), mut tick: ResMut<TickValue>| tick.0.take(),
                    OutputDelivery::broadcast(),
                )
            })
        }
    }

    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<TickValue>();

    let source = TickSource.cache().broadcast();
    let deep_branch = source
        .clone()
        .map(|In(v): In<i32>| v + 1)
        .map(|In(v): In<i32>| v + 1)
        .map(|In(v): In<i32>| v + 1)
        .map(|In(v): In<i32>| v + 1)
        .map(|In(v): In<i32>| v * 100)
        .cache()
        .broadcast();
    // Keep `deep_branch` active at levels 1..=6 so that switching from `source` (shallow adapter at
    // level 1, owner at level 3) to `deep_branch` (deep adapter at level 7, owner promoted 3 -> 8)
    // exercises Phase 2 mid-pass level promotion in a single topological pass!
    let deep_claim = deep_branch.clone().register(app.world_mut());

    let seen = Arc::new(Mutex::new(Vec::new()));
    let claim = source
        .clone()
        .map(|In(v): In<i32>| v >= 2)
        .switch({
            let source = source.clone();
            let deep_branch = deep_branch.clone();
            move |In(deep): In<bool>| {
                if deep {
                    deep_branch.clone().boxed_replayable()
                } else {
                    source.clone().boxed_replayable()
                }
            }
        })
        .map({
            let seen = Arc::clone(&seen);
            move |In(v): In<i32>| {
                seen.lock().unwrap_or_else(|e| e.into_inner()).push(v);
            }
        })
        .register(app.world_mut());

    // Initial tick (1): shallow branch emits 1 (while deep_branch warms to 500).
    app.update();
    assert_eq!(*seen.lock().unwrap_or_else(|e| e.into_inner()), vec![1]);

    // Tick 2: shallow adapter runs at level 1 (queuing 2), manager runs at level 2 (switching to
    // deep_branch whose pre-frame cache is 500, and promoting owner from level 3 to level 8),
    // deep_branch computes (2 + 4) * 100 = 600 across levels 1..=7, and owner skips stale bucket 3
    // and runs ONCE at level 8 emitting ONLY 600 (never stale 2 or stale 500!).
    app.world_mut().resource_mut::<TickValue>().0 = Some(2);
    app.update();
    assert_eq!(*seen.lock().unwrap_or_else(|e| e.into_inner()), vec![1, 600]);

    claim.release(app.world_mut());
    deep_claim.release(app.world_mut());
}

#[test]
fn scalar_dynamic_keep_current_replays_after_clear_and_cancels_warming_transient() {
    type Update = DynamicManagerUpdate<ScalarSlotCommand<ReplayableBroadcast<i32>>, i32>;

    #[derive(Resource, Default)]
    struct WarmingPulse(Option<i32>);

    struct ReadyConst(i32);
    impl Signal for ReadyConst {
        type Item = i32;
        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
            let initial = self.0;
            ScalarSignalMaterializer::new(move |registrar| {
                registrar.replayable_signal_source(initial, |In(())| None::<i32>, OutputDelivery::broadcast())
            })
        }
    }

    struct WarmingSignal;
    impl Signal for WarmingSignal {
        type Item = i32;
        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<i32> {
            ScalarSignalMaterializer::new(|registrar| {
                registrar.replayable_signal_entity(
                    (),
                    |In(()), mut pulse: ResMut<WarmingPulse>| pulse.0.take(),
                    ReplayInitialization::Warming,
                    OutputDelivery::broadcast(),
                )
            })
        }
    }

    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());
    app.init_resource::<WarmingPulse>();

    let ready_a = ReadyConst(10).cache().broadcast();
    let warming_b = WarmingSignal.cache().broadcast();

    let updates: Arc<Mutex<VecDeque<Update>>> = Arc::new(Mutex::new(VecDeque::from([
        // Frame 1: select A -> emits 10.
        Update::new(
            [ScalarSlotCommand::Select {
                signal: ready_a.clone(),
            }],
            DynamicEmissions::None,
        ),
        // Frame 2: [Clear, Select(A)] -> must replay 10 again (not silently suppress via
        // KeepCurrent!).
        Update::new(
            [
                ScalarSlotCommand::Clear,
                ScalarSlotCommand::Select {
                    signal: ready_a.clone(),
                },
            ],
            DynamicEmissions::None,
        ),
        // Frame 3: Select(warming_b) -> A stays active while B warms.
        Update::new(
            [ScalarSlotCommand::Select {
                signal: warming_b.clone(),
            }],
            DynamicEmissions::None,
        ),
        // Frame 4: [Clear, Select(A)] -> must cancel warming_b and replay A (10)!
        Update::new(
            [
                ScalarSlotCommand::Clear,
                ScalarSlotCommand::Select {
                    signal: ready_a.clone(),
                },
            ],
            DynamicEmissions::None,
        ),
    ])));

    struct CustomOperator {
        updates: Arc<Mutex<VecDeque<Update>>>,
    }

    impl Signal for CustomOperator {
        type Item = Vec<i32>;
        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Vec<i32>> {
            let updates = self.updates;
            ScalarSignalMaterializer::new(move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(())| updates.lock().unwrap_or_else(|e| e.into_inner()).pop_front(),
                    OutputDelivery::move_only(),
                )?;
                registrar.dynamic_operator(
                    DynamicSignalOperatorSpec::scalar(manager)
                        .replay_scalar::<i32>()
                        .adapter(|(), value| DynamicEmissions::One(value))
                        .scalar_output(|batch: Vec<i32>| (!batch.is_empty()).then_some(batch))
                        .same_signal_identity(SameSignalIdentity::KeepCurrent),
                )
            })
        }
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let claim = CustomOperator { updates }
        .map({
            let seen = Arc::clone(&seen);
            move |In(batch): In<Vec<i32>>| {
                seen.lock().unwrap_or_else(|e| e.into_inner()).push(batch);
            }
        })
        .register(app.world_mut());

    // Frame 1: initial Select(A) -> [10]
    app.update();
    assert_eq!(*seen.lock().unwrap_or_else(|e| e.into_inner()), vec![vec![10]]);

    // Frame 2: [Clear, Select(A)] -> [10] replayed!
    app.update();
    assert_eq!(
        *seen.lock().unwrap_or_else(|e| e.into_inner()),
        vec![vec![10], vec![10]]
    );

    // Frame 3: Select(warming_b) -> nothing emitted yet while warming_b warms.
    app.update();
    assert_eq!(
        *seen.lock().unwrap_or_else(|e| e.into_inner()),
        vec![vec![10], vec![10]]
    );

    // Frame 4: [Clear, Select(A)] -> warming_b cancelled, A replayed -> [10].
    app.update();
    assert_eq!(
        *seen.lock().unwrap_or_else(|e| e.into_inner()),
        vec![vec![10], vec![10], vec![10]]
    );

    // Frame 5: pulsing warming_b now must NOT leak through because warming_b was cancelled!
    app.world_mut().resource_mut::<WarmingPulse>().0 = Some(999);
    app.update();
    assert_eq!(
        *seen.lock().unwrap_or_else(|e| e.into_inner()),
        vec![vec![10], vec![10], vec![10]]
    );

    claim.release(app.world_mut());
}

#[test]
fn unretained_dynamic_combinators_reset_cleanly_across_deactivation_and_reactivation() {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    // 1. Unretained MapSignal + Flatten upstream of a retained `.broadcast()` terminal.
    let inner_a = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let inner_b = MutableVec::builder().values([10]).spawn(app.world_mut());
    let outer = MutableVec::builder()
        .values([inner_a.signal_vec().broadcast(), inner_b.signal_vec().broadcast()])
        .spawn(app.world_mut());

    let seen_state = Arc::new(Mutex::new(Vec::new()));
    let retained_terminal = outer
        .signal_vec()
        .flatten()
        .for_each({
            let seen_state = Arc::clone(&seen_state);
            move |In(diffs): In<Vec<VecDiff<i32>>>| {
                let mut state = seen_state.lock().unwrap_or_else(|e| e.into_inner());
                for diff in diffs {
                    diff.apply_to_vec(&mut state);
                }
            }
        })
        .broadcast();

    let claim1 = retained_terminal.clone().register(app.world_mut());
    app.update();
    assert_eq!(*seen_state.lock().unwrap_or_else(|e| e.into_inner()), vec![1, 2, 10]);

    // Release 1 -> 0 while mutating both outer and inner MutableVecs during inactivity.
    claim1.release(app.world_mut());
    seen_state.lock().unwrap_or_else(|e| e.into_inner()).clear();
    inner_a.write(app.world_mut()).push(3);
    outer.write(app.world_mut()).pop();
    app.update();
    assert!(
        seen_state.lock().unwrap_or_else(|e| e.into_inner()).is_empty(),
        "inactive terminal must not receive updates"
    );

    // Reactivate 0 -> 1: unretained Flatten + VecReplayCache must replay fresh [1, 2, 3] cleanly!
    let claim2 = retained_terminal.register(app.world_mut());
    app.update();
    assert_eq!(*seen_state.lock().unwrap_or_else(|e| e.into_inner()), vec![1, 2, 3]);
    claim2.release(app.world_mut());
}

#[cfg(feature = "builder")]
#[test]
fn flatten_drives_builder_children_and_multi_builder_spawn_on_entity_preserves_blocks() {
    #[derive(Component, Debug, PartialEq, Eq)]
    struct Tag(&'static str, i32);

    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    let group_1 = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
    let group_2 = MutableVec::builder().values([10]).spawn(app.world_mut());
    let groups = MutableVec::builder()
        .values([group_1.signal_vec().broadcast(), group_2.signal_vec().broadcast()])
        .spawn(app.world_mut());

    let extra = MutableVec::builder().values([100, 200]).spawn(app.world_mut());

    let parent = app.world_mut().spawn_empty().id();

    let builder_a = Builder::new()
        .child(Builder::new().insert(Tag("static_a", 0)))
        .children_signal_vec(
            groups
                .signal_vec()
                .flatten()
                .map_in(|v| Builder::new().insert(Tag("flat", v))),
        );

    let builder_b = Builder::new()
        .child(Builder::new().insert(Tag("static_b", 99)))
        .children_signal_vec(extra.signal_vec().map_in(|v| Builder::new().insert(Tag("extra", v))));

    builder_a.spawn_on_entity(app.world_mut(), parent).expect("builder_a");
    builder_b.spawn_on_entity(app.world_mut(), parent).expect("builder_b");

    app.update();

    let read_children = |world: &World| -> Vec<(&'static str, i32)> {
        world
            .get::<Children>(parent)
            .map(|children| {
                children
                    .iter()
                    .filter_map(|child| world.get::<Tag>(child).map(|tag| (tag.0, tag.1)))
                    .collect()
            })
            .unwrap_or_default()
    };

    assert_eq!(
        read_children(app.world()),
        vec![
            ("static_a", 0),
            ("flat", 1),
            ("flat", 2),
            ("flat", 10),
            ("static_b", 99),
            ("extra", 100),
            ("extra", 200),
        ]
    );

    // Mutate both flattened inner/outer vectors AND the second builder's vector!
    group_1.write(app.world_mut()).insert(1, 5);
    extra.write(app.world_mut()).push(300);
    app.update();

    assert_eq!(
        read_children(app.world()),
        vec![
            ("static_a", 0),
            ("flat", 1),
            ("flat", 5),
            ("flat", 2),
            ("flat", 10),
            ("static_b", 99),
            ("extra", 100),
            ("extra", 200),
            ("extra", 300),
        ]
    );

    // Despawn one dynamic child externally, then pop/remove from group_1 without panicking!
    let external_victim = app
        .world()
        .get::<Children>(parent)
        .and_then(|children| children.get(2).copied())
        .expect("flat child 5");
    app.world_mut().despawn(external_victim);
    group_1.write(app.world_mut()).remove(1);
    app.update();

    assert_eq!(
        read_children(app.world()),
        vec![
            ("static_a", 0),
            ("flat", 1),
            ("flat", 2),
            ("flat", 10),
            ("static_b", 99),
            ("extra", 100),
            ("extra", 200),
            ("extra", 300),
        ]
    );
}
