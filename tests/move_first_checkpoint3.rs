//! Integration coverage for Move-First Checkpoint 3:
//! scalar `Clone` bound relaxation, end-to-end `NonClone` move-only scalar chains,
//! `OutputDelivery::move_only()` defaults on ordinary scalar combinators and
//! `DynamicSignalOperatorSpec`, and superseded owned-buffer drop accounting.

extern crate alloc;

use alloc::{boxed::Box, format, string::String, sync::Arc, vec, vec::Vec};
use bevy_app::{App, Last};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel, system::SystemIdMarker};
use core::{
    ops::{Add, Not},
    sync::atomic::{AtomicUsize, Ordering},
};
use jonmo::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, OutputDelivery, SameSignalIdentity,
    ScalarSignalMaterializer, ScalarSlotCommand, SignalRegistrationError, prelude::*, signal,
};
use std::sync::Mutex;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
struct Producer;

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<Producer>());
    app
}

fn system_entity_count(world: &mut World) -> usize {
    world
        .query_filtered::<Entity, With<SystemIdMarker>>()
        .iter(world)
        .count()
}

/// Strictly move-only (`!Clone`) scalar payload with drop accounting.
#[derive(Debug)]
struct NonClone {
    label: String,
    drops: Arc<AtomicUsize>,
}

impl NonClone {
    fn new(label: impl Into<String>, drops: &Arc<AtomicUsize>) -> Self {
        Self {
            label: label.into(),
            drops: Arc::clone(drops),
        }
    }
}

impl PartialEq for NonClone {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label
    }
}

impl Eq for NonClone {}

impl Drop for NonClone {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

/// Strictly move-only (`!Clone`) flag whose [`Not::Output`] is also [`NonClone`].
#[derive(Debug)]
struct NonCloneFlag {
    active: bool,
    drops: Arc<AtomicUsize>,
}

impl NonCloneFlag {
    fn new(active: bool, drops: &Arc<AtomicUsize>) -> Self {
        Self {
            active,
            drops: Arc::clone(drops),
        }
    }
}

impl Drop for NonCloneFlag {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Not for NonCloneFlag {
    type Output = NonClone;

    fn not(self) -> Self::Output {
        let inverted = !self.active;
        NonClone::new(format!("not:{inverted}"), &self.drops)
    }
}

/// Move-only (`!Clone`) entity wrapper for `from_entity`, `from_ancestor`, and `from_parent`.
struct NonCloneIntoEntity(Entity);

impl From<NonCloneIntoEntity> for Entity {
    fn from(value: NonCloneIntoEntity) -> Self {
        value.0
    }
}

/// Strictly move-only (`!Clone`) component for `has_component`.
#[derive(Component)]
struct NonCloneComponent(#[expect(dead_code)] NonClone);

#[derive(Debug)]
struct CloneProbe {
    value: i32,
    clones: Arc<AtomicUsize>,
}

impl CloneProbe {
    fn new(value: i32, clones: &Arc<AtomicUsize>) -> Self {
        Self {
            value,
            clones: Arc::clone(clones),
        }
    }
}

impl Clone for CloneProbe {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            value: self.value,
            clones: Arc::clone(&self.clones),
        }
    }
}

impl PartialEq for CloneProbe {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for CloneProbe {}

#[derive(Clone, Debug)]
struct AddToNonClone {
    value: i32,
    drops: Arc<AtomicUsize>,
}

impl Add for AddToNonClone {
    type Output = NonClone;

    fn add(self, rhs: Self) -> Self::Output {
        NonClone::new(format!("sum:{}", self.value + rhs.value), &self.drops)
    }
}

struct CustomMoveMap<Upstream> {
    upstream: Upstream,
    prefix: &'static str,
}

impl<Upstream> Signal for CustomMoveMap<Upstream>
where
    Upstream: Signal<Item = NonClone>,
{
    type Item = NonClone;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        let Self { upstream, prefix } = *self;
        let upstream = Box::new(upstream).into_signal_materializer();
        ScalarSignalMaterializer::new(move |registrar| {
            let upstream = registrar.materialize_signal(upstream)?;
            registrar.signal_entity(
                upstream,
                move |In(item): In<NonClone>| {
                    let drops = Arc::clone(&item.drops);
                    let next = format!("{prefix}:{}", item.label);
                    drop(item);
                    Some(NonClone::new(next, &drops))
                },
                OutputDelivery::move_only(),
            )
        })
    }
}

#[test]
fn non_clone_sources_entity_constructors_and_map_variants_flow_end_to_end() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));
    let once_observed = Arc::new(Mutex::new(Vec::<String>::new()));
    let sys_observed = Arc::new(Mutex::new(Vec::<String>::new()));
    let fn_observed = Arc::new(Mutex::new(Vec::<String>::new()));

    // 1. `once(NonClone)` -> `map` -> `map_in` -> `map_in_ref` (producing a fresh `NonClone`).
    let once_claim = signal::once(NonClone::new("seed", &drops))
        .map(|In(mut item): In<NonClone>| {
            item.label.push_str(":map");
            item
        })
        .map_in(|mut item: NonClone| {
            item.label.push_str(":map_in");
            item
        })
        .map_in_ref({
            let drops = Arc::clone(&drops);
            move |item: &NonClone| NonClone::new(format!("{}:map_in_ref", item.label), &drops)
        })
        .map({
            let once_observed = Arc::clone(&once_observed);
            move |In(item): In<NonClone>| {
                once_observed.lock().expect("once lock").push(item.label.clone());
            }
        })
        .register(app.world_mut());

    // 2. `from_system` + `from_function` producing `NonClone`.
    let sys_claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut step: Local<usize>| {
            *step += 1;
            (*step <= 2).then(|| NonClone::new(format!("sys:{}", *step), &drops))
        }
    })
    .map({
        let sys_observed = Arc::clone(&sys_observed);
        move |In(item): In<NonClone>| {
            sys_observed.lock().expect("sys lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    let mut fn_step = 0usize;
    let fn_claim = signal::from_function({
        let drops = Arc::clone(&drops);
        move || {
            fn_step += 1;
            NonClone::new(format!("fn:{fn_step}"), &drops)
        }
    })
    .take(2)
    .map({
        let fn_observed = Arc::clone(&fn_observed);
        move |In(item): In<NonClone>| {
            fn_observed.lock().expect("fn lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    // 3. `from_entity`, `from_parent`, `from_ancestor` with a move-only `Into<Entity>` source,
    //    including `1 -> 0 -> 1` reactivation when retained by a downstream `.broadcast()`.
    let grandparent = app.world_mut().spawn_empty().id();
    let parent = app.world_mut().spawn(ChildOf(grandparent)).id();
    let child = app.world_mut().spawn(ChildOf(parent)).id();

    let entity_entities = Arc::new(Mutex::new(Vec::<(Entity, Entity, Entity)>::new()));
    let retained_parent = signal::from_parent(NonCloneIntoEntity(child)).broadcast();
    let hold_broadcast = retained_parent.clone();

    let hierarchy_claim = signal::from_entity(NonCloneIntoEntity(child))
        .zip(retained_parent)
        .zip(signal::from_ancestor(NonCloneIntoEntity(child), 2))
        .map_in({
            let drops = Arc::clone(&drops);
            move |((self_entity, parent_entity), grandparent_entity)| {
                (
                    (self_entity, parent_entity, grandparent_entity),
                    NonClone::new("hierarchy", &drops),
                )
            }
        })
        .map({
            let entity_entities = Arc::clone(&entity_entities);
            move |In((triple, token)): In<((Entity, Entity, Entity), NonClone)>| {
                assert_eq!(token.label, "hierarchy");
                entity_entities.lock().expect("hierarchy lock").push(triple);
            }
        })
        .register(app.world_mut());

    app.update();
    app.update();
    app.update();

    assert_eq!(
        *once_observed.lock().expect("once lock"),
        vec!["seed:map:map_in:map_in_ref"]
    );
    assert_eq!(*sys_observed.lock().expect("sys lock"), vec!["sys:1", "sys:2"]);
    assert_eq!(*fn_observed.lock().expect("fn lock"), vec!["fn:1", "fn:2"]);
    assert_eq!(
        *entity_entities.lock().expect("hierarchy lock"),
        vec![
            (child, parent, grandparent),
            (child, parent, grandparent),
            (child, parent, grandparent),
        ]
    );

    // Release `hierarchy_claim` while `hold_broadcast` stays alive (`1 -> 0` deactivation),
    // then re-attach (`0 -> 1` reactivation) to verify `Local<Option<Entity>>` survives.
    hierarchy_claim.release(app.world_mut());
    let reactivated_parent = Arc::new(Mutex::new(Vec::<Entity>::new()));
    let reactivated_claim = hold_broadcast
        .map({
            let reactivated_parent = Arc::clone(&reactivated_parent);
            move |In(entity): In<Entity>| {
                reactivated_parent.lock().expect("parent lock").push(entity);
            }
        })
        .register(app.world_mut());
    app.update();
    assert_eq!(*reactivated_parent.lock().expect("parent lock"), vec![parent]);

    once_claim.release(app.world_mut());
    sys_claim.release(app.world_mut());
    fn_claim.release(app.world_mut());
    reactivated_claim.release(app.world_mut());

    // Total `NonClone` instances created across all 4 frames:
    // - `once`: 1 ("seed") + 1 ("seed:...:map_in_ref") = 2
    // - `from_system`: 2 ("sys:1", "sys:2")
    // - `from_function`: 4 ("fn:1", "fn:2", plus "fn:3" and "fn:4" dropped by `.take(2)`)
    // - `hierarchy`: 3 ("hierarchy" on frames 1..=3 before release)
    assert_eq!(drops.load(Ordering::SeqCst), 11);
}

#[test]
fn non_clone_take_skip_first_eq_neq_and_not_flow_end_to_end() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));

    let skipped_out = Arc::new(Mutex::new(Vec::<String>::new()));
    let skip_take_claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut step: Local<usize>| {
            *step += 1;
            Some(NonClone::new(format!("item:{}", *step), &drops))
        }
    })
    .skip(2)
    .take(2)
    .map({
        let skipped_out = Arc::clone(&skipped_out);
        move |In(item): In<NonClone>| {
            skipped_out.lock().expect("skip lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    let first_out = Arc::new(Mutex::new(Vec::<String>::new()));
    let first_claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut step: Local<usize>| {
            *step += 1;
            Some(NonClone::new(format!("first:{}", *step), &drops))
        }
    })
    .first()
    .map({
        let first_out = Arc::clone(&first_out);
        move |In(item): In<NonClone>| {
            first_out.lock().expect("first lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    let eq_out = Arc::new(Mutex::new(Vec::<bool>::new()));
    let eq_claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut step: Local<usize>| {
            *step += 1;
            Some(NonClone::new(format!("cmp:{}", *step), &drops))
        }
    })
    .eq(NonClone::new("cmp:2", &drops))
    .map({
        let eq_out = Arc::clone(&eq_out);
        move |In(equal): In<bool>| {
            eq_out.lock().expect("eq lock").push(equal);
        }
    })
    .register(app.world_mut());

    let neq_out = Arc::new(Mutex::new(Vec::<bool>::new()));
    let neq_claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut step: Local<usize>| {
            *step += 1;
            Some(NonClone::new(format!("cmp:{}", *step), &drops))
        }
    })
    .neq(NonClone::new("cmp:2", &drops))
    .map({
        let neq_out = Arc::clone(&neq_out);
        move |In(not_equal): In<bool>| {
            neq_out.lock().expect("neq lock").push(not_equal);
        }
    })
    .register(app.world_mut());

    let not_out = Arc::new(Mutex::new(Vec::<String>::new()));
    let not_claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut flag: Local<bool>| {
            let current = *flag;
            *flag = !*flag;
            NonCloneFlag::new(current, &drops)
        }
    })
    .not()
    .take(3)
    .map({
        let not_out = Arc::clone(&not_out);
        move |In(item): In<NonClone>| {
            not_out.lock().expect("not lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    for _ in 0..5 {
        app.update();
    }

    assert_eq!(*skipped_out.lock().expect("skip lock"), vec!["item:3", "item:4"]);
    assert_eq!(*first_out.lock().expect("first lock"), vec!["first:1"]);
    assert_eq!(*eq_out.lock().expect("eq lock"), vec![false, true, false, false, false]);
    assert_eq!(*neq_out.lock().expect("neq lock"), vec![true, false, true, true, true]);
    assert_eq!(
        *not_out.lock().expect("not lock"),
        vec!["not:true", "not:false", "not:true"]
    );

    skip_take_claim.release(app.world_mut());
    first_claim.release(app.world_mut());
    eq_claim.release(app.world_mut());
    neq_claim.release(app.world_mut());
    not_claim.release(app.world_mut());
}

#[test]
fn non_clone_bool_and_option_projections_flow_end_to_end() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));
    let log = Arc::new(Mutex::new(Vec::<String>::new()));

    let bool_source = || {
        signal::from_system(|In(()), mut step: Local<usize>| {
            *step += 1;
            (*step <= 2).then_some(*step == 1)
        })
    };

    let option_source = |drops: Arc<AtomicUsize>| {
        signal::from_system(move |In(()), mut step: Local<usize>| {
            *step += 1;
            match *step {
                1 => Some(Some(NonClone::new("opt:1", &drops))),
                2 => Some(None),
                _ => None,
            }
        })
    };

    let claims = vec![
        bool_source()
            .map_bool(
                {
                    let drops = Arc::clone(&drops);
                    move |In(())| NonClone::new("map_bool:true", &drops)
                },
                {
                    let drops = Arc::clone(&drops);
                    move |In(())| NonClone::new("map_bool:false", &drops)
                },
            )
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<NonClone>| log.lock().expect("log").push(item.label.clone())
            })
            .register(app.world_mut()),
        bool_source()
            .map_bool_in(
                {
                    let drops = Arc::clone(&drops);
                    move || NonClone::new("map_bool_in:true", &drops)
                },
                {
                    let drops = Arc::clone(&drops);
                    move || NonClone::new("map_bool_in:false", &drops)
                },
            )
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<NonClone>| log.lock().expect("log").push(item.label.clone())
            })
            .register(app.world_mut()),
        bool_source()
            .map_true({
                let drops = Arc::clone(&drops);
                move |In(())| NonClone::new("map_true", &drops)
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => format!("some:{}", value.label),
                        None => String::from("none:map_true"),
                    });
                }
            })
            .register(app.world_mut()),
        bool_source()
            .map_true_in({
                let drops = Arc::clone(&drops);
                move || NonClone::new("map_true_in", &drops)
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => format!("some:{}", value.label),
                        None => String::from("none:map_true_in"),
                    });
                }
            })
            .register(app.world_mut()),
        bool_source()
            .map_false({
                let drops = Arc::clone(&drops);
                move |In(())| NonClone::new("map_false", &drops)
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => format!("some:{}", value.label),
                        None => String::from("none:map_false"),
                    });
                }
            })
            .register(app.world_mut()),
        bool_source()
            .map_false_in({
                let drops = Arc::clone(&drops);
                move || NonClone::new("map_false_in", &drops)
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => format!("some:{}", value.label),
                        None => String::from("none:map_false_in"),
                    });
                }
            })
            .register(app.world_mut()),
        option_source(Arc::clone(&drops))
            .map_option(
                |In(mut item): In<NonClone>| {
                    item.label.push_str(":map_option_some");
                    item
                },
                {
                    let drops = Arc::clone(&drops);
                    move |In(())| NonClone::new("map_option_none", &drops)
                },
            )
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<NonClone>| log.lock().expect("log").push(item.label.clone())
            })
            .register(app.world_mut()),
        option_source(Arc::clone(&drops))
            .map_option_in(
                |mut item: NonClone| {
                    item.label.push_str(":map_option_in_some");
                    item
                },
                {
                    let drops = Arc::clone(&drops);
                    move || NonClone::new("map_option_in_none", &drops)
                },
            )
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<NonClone>| log.lock().expect("log").push(item.label.clone())
            })
            .register(app.world_mut()),
        option_source(Arc::clone(&drops))
            .map_some(|In(mut item): In<NonClone>| {
                item.label.push_str(":map_some");
                item
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => value.label.clone(),
                        None => String::from("none:map_some"),
                    });
                }
            })
            .register(app.world_mut()),
        option_source(Arc::clone(&drops))
            .map_some_in(|mut item: NonClone| {
                item.label.push_str(":map_some_in");
                item
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => value.label.clone(),
                        None => String::from("none:map_some_in"),
                    });
                }
            })
            .register(app.world_mut()),
        option_source(Arc::clone(&drops))
            .map_none({
                let drops = Arc::clone(&drops);
                move |In(())| NonClone::new("map_none_hit", &drops)
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => value.label.clone(),
                        None => String::from("none:map_none"),
                    });
                }
            })
            .register(app.world_mut()),
        option_source(Arc::clone(&drops))
            .map_none_in({
                let drops = Arc::clone(&drops);
                move || NonClone::new("map_none_in_hit", &drops)
            })
            .map({
                let log = Arc::clone(&log);
                move |In(item): In<Option<NonClone>>| {
                    log.lock().expect("log").push(match item {
                        Some(value) => value.label.clone(),
                        None => String::from("none:map_none_in"),
                    });
                }
            })
            .register(app.world_mut()),
    ];

    app.update();
    app.update();
    app.update();

    let entries = log.lock().expect("log").clone();
    for expected in [
        "map_bool:true",
        "map_bool:false",
        "map_bool_in:true",
        "map_bool_in:false",
        "some:map_true",
        "none:map_true",
        "some:map_true_in",
        "none:map_true_in",
        "none:map_false",
        "some:map_false",
        "none:map_false_in",
        "some:map_false_in",
        "opt:1:map_option_some",
        "map_option_none",
        "opt:1:map_option_in_some",
        "map_option_in_none",
        "opt:1:map_some",
        "none:map_some",
        "opt:1:map_some_in",
        "none:map_some_in",
        "none:map_none",
        "map_none_hit",
        "none:map_none_in",
        "map_none_in_hit",
    ] {
        assert!(
            entries.iter().any(|actual| actual == expected),
            "missing expected projection output `{expected}` in {entries:?}"
        );
    }

    for claim in claims {
        claim.release(app.world_mut());
    }
}

#[cfg(feature = "time")]
#[test]
fn non_clone_throttle_moves_accepted_items_and_drops_throttled_items() {
    use bevy_time::Time;
    use core::time::Duration;

    let mut app = test_app();
    app.init_resource::<Time>();
    let drops = Arc::new(AtomicUsize::new(0));
    let delivered = Arc::new(Mutex::new(Vec::<String>::new()));

    let claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut step: Local<usize>| {
            *step += 1;
            Some(NonClone::new(format!("tick:{}", *step), &drops))
        }
    })
    .throttle(Duration::from_millis(100))
    .map({
        let delivered = Arc::clone(&delivered);
        move |In(item): In<NonClone>| {
            delivered.lock().expect("delivered lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    // Frame 1: initial item passes immediately and starts the 100ms timer.
    app.update();
    assert_eq!(*delivered.lock().expect("delivered lock"), vec!["tick:1"]);
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    // Frame 2 (+40ms): throttled; `tick:2` is dropped immediately.
    app.world_mut()
        .resource_mut::<Time>()
        .advance_by(Duration::from_millis(40));
    app.update();
    assert_eq!(*delivered.lock().expect("delivered lock"), vec!["tick:1"]);
    assert_eq!(drops.load(Ordering::SeqCst), 2);

    // Frame 3 (+70ms, total 110ms): timer finishes; `tick:3` moves downstream!
    app.world_mut()
        .resource_mut::<Time>()
        .advance_by(Duration::from_millis(70));
    app.update();
    assert_eq!(*delivered.lock().expect("delivered lock"), vec!["tick:1", "tick:3"]);
    assert_eq!(drops.load(Ordering::SeqCst), 3);

    claim.release(app.world_mut());
}

#[cfg(feature = "tracing")]
#[test]
fn non_clone_debug_logs_by_ref_and_forwards_by_move() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));
    let delivered = Arc::new(Mutex::new(Vec::<String>::new()));

    let claim = signal::once(NonClone::new("traced", &drops))
        .debug()
        .map({
            let delivered = Arc::clone(&delivered);
            move |In(item): In<NonClone>| {
                delivered.lock().expect("delivered lock").push(item.label.clone());
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(*delivered.lock().expect("delivered lock"), vec!["traced"]);
    claim.release(app.world_mut());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn non_clone_boxed_either_scheduled_superseded_buffer_try_register_and_custom_signal() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));
    let delivered = Arc::new(Mutex::new(Vec::<String>::new()));

    // 1. Heterogeneous `BoxedSignal<NonClone>` and `SignalEither<..., ...>` + custom `Signal`.
    let make_branch = |left: bool, drops: &Arc<AtomicUsize>| {
        if left {
            signal::once(NonClone::new("either:left", drops)).left_either()
        } else {
            CustomMoveMap {
                upstream: signal::once(NonClone::new("either:right", drops)),
                prefix: "custom",
            }
            .right_either()
        }
    };

    let boxed_signals: Vec<BoxedSignal<NonClone>> =
        vec![make_branch(true, &drops).boxed(), make_branch(false, &drops).boxed()];

    let mut branch_claims = Vec::new();
    for branch in boxed_signals {
        let claim = branch
            .map({
                let delivered = Arc::clone(&delivered);
                move |In(item): In<NonClone>| {
                    delivered.lock().expect("delivered lock").push(item.label.clone());
                }
            })
            .try_register(app.world_mut())
            .expect("valid move-only boxed signal must register cleanly");
        branch_claims.push(claim);
    }

    app.update();
    assert_eq!(
        *delivered.lock().expect("delivered lock"),
        vec!["either:left", "custom:either:right"]
    );
    for claim in branch_claims {
        claim.release(app.world_mut());
    }
    // "either:left" (1) + "either:right" (1) + "custom:either:right" (1) = 3 drops.
    assert_eq!(drops.load(Ordering::SeqCst), 3);

    // 2a. Multi-input scalar burst coalescing with `NonClone`:
    //     `Producer` stages 4 ticks (`1, 2, 3, 4`) before `Last` runs.
    //     In `Last`, `stage` runs for all 4 queued inputs, producing `Some("burst:1")`,
    //     `Some("burst:2")`, `Some("burst:3")`, and `None`.
    //     Scalar burst coalescing must drop superseded `"burst:1"` and `"burst:2"`, preserve
    //     `"burst:3"` across the trailing `None`, and move only `"burst:3"` into the downstream
    //     consumer!
    drops.store(0, Ordering::SeqCst);
    delivered.lock().expect("delivered lock").clear();
    let stage_calls = Arc::new(Mutex::new(Vec::<usize>::new()));

    let burst_claim = signal::from_system(|In(()), mut poll: Local<usize>| {
        *poll += 1;
        (*poll <= 4).then_some(*poll)
    })
    .schedule::<Producer>()
    .map({
        let drops = Arc::clone(&drops);
        let stage_calls = Arc::clone(&stage_calls);
        move |In(poll): In<usize>| {
            stage_calls.lock().expect("stage lock").push(poll);
            (poll <= 3).then(|| NonClone::new(format!("burst:{poll}"), &drops))
        }
    })
    .schedule::<Last>()
    .map({
        let delivered = Arc::clone(&delivered);
        move |In(item): In<NonClone>| {
            delivered.lock().expect("delivered lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    for _ in 0..4 {
        app.world_mut().run_schedule(Producer);
    }
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    app.world_mut().run_schedule(Last);
    assert_eq!(*stage_calls.lock().expect("stage lock"), vec![1, 2, 3, 4]);
    assert_eq!(*delivered.lock().expect("delivered lock"), vec!["burst:3"]);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
    burst_claim.release(app.world_mut());

    // 2b. Releasing a consumer before drain drops all queued `NonClone` inputs without leaking.
    drops.store(0, Ordering::SeqCst);
    let unreleased_queue_claim = signal::from_system({
        let drops = Arc::clone(&drops);
        move |In(()), mut poll: Local<usize>| {
            *poll += 1;
            Some(NonClone::new(format!("queued:{}", *poll), &drops))
        }
    })
    .schedule::<Producer>()
    .map(|In(_item): In<NonClone>| {})
    .schedule::<Last>()
    .register(app.world_mut());

    app.world_mut().run_schedule(Producer);
    app.world_mut().run_schedule(Producer);
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    unreleased_queue_claim.release(app.world_mut());
    assert_eq!(drops.load(Ordering::SeqCst), 2);

    // 3. Direct `.register()` and `.try_register()` on `Signal<Item = NonClone>` (with no trailing
    //    `()` map), `signal::option` with `NonClone` (`Some` and `None` without implicit cache
    //    entities), `has_component::<NonCloneComponent>()`, and `NonClone`-driven `.switch(...)`.
    drops.store(0, Ordering::SeqCst);
    let opt_none_claim = signal::option::<BoxedSignal<NonClone>>(None).register(app.world_mut());
    assert_eq!(
        system_entity_count(app.world_mut()),
        1,
        "linear `signal::option(None)` must materialize only `once(None)` with no implicit cache entity"
    );
    opt_none_claim.release(app.world_mut());
    assert_eq!(system_entity_count(app.world_mut()), 0);

    let opt_some_out = Arc::new(Mutex::new(Vec::<Option<String>>::new()));
    let opt_some_claim = signal::option(Some(signal::once(NonClone::new("opt:some", &drops))))
        .map({
            let opt_some_out = Arc::clone(&opt_some_out);
            move |In(item): In<Option<NonClone>>| {
                opt_some_out
                    .lock()
                    .expect("opt_some lock")
                    .push(item.map(|value| value.label.clone()));
            }
        })
        .register(app.world_mut());
    assert_eq!(
        system_entity_count(app.world_mut()),
        3,
        "linear `signal::option(Some(once)).map(...)` must materialize once + map_in(Some) + consumer with no cache entity"
    );

    let direct_reg = signal::once(NonClone::new("direct:register", &drops)).register(app.world_mut());
    let direct_try_reg = signal::once(NonClone::new("direct:try_register", &drops))
        .boxed()
        .try_register(app.world_mut())
        .expect("direct Signal<Item = NonClone> try_register must succeed");
    let tagged_entity = app
        .world_mut()
        .spawn(NonCloneComponent(NonClone::new("component", &drops)))
        .id();
    let has_comp_out = Arc::new(Mutex::new(Vec::<bool>::new()));
    let has_comp_claim = signal::from_entity(tagged_entity)
        .has_component::<NonCloneComponent>()
        .map({
            let has_comp_out = Arc::clone(&has_comp_out);
            move |In(present): In<bool>| {
                has_comp_out.lock().expect("has_comp lock").push(present);
            }
        })
        .register(app.world_mut());
    let switched_out = Arc::new(Mutex::new(Vec::<usize>::new()));
    let switch_claim = signal::once(NonClone::new("driver", &drops))
        .switch(|In(driver): In<NonClone>| signal::once(driver.label.len()).cache())
        .map({
            let switched_out = Arc::clone(&switched_out);
            move |In(len): In<usize>| {
                switched_out.lock().expect("switch lock").push(len);
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(
        *opt_some_out.lock().expect("opt_some lock"),
        vec![Some(String::from("opt:some"))]
    );
    assert_eq!(*has_comp_out.lock().expect("has_comp lock"), vec![true]);
    assert_eq!(*switched_out.lock().expect("switch lock"), vec![6]);
    // "opt:some" (1) + "direct:register" (1) + "direct:try_register" (1) + "driver" (1) = 4
    // dropped; "component" is still alive on `tagged_entity`.
    assert_eq!(drops.load(Ordering::SeqCst), 4);

    opt_some_claim.release(app.world_mut());
    direct_reg.release(app.world_mut());
    direct_try_reg.release(app.world_mut());
    has_comp_claim.release(app.world_mut());
    switch_claim.release(app.world_mut());
    app.world_mut().despawn(tagged_entity);
    assert_eq!(drops.load(Ordering::SeqCst), 5);
}

#[cfg(feature = "builder")]
#[test]
fn non_clone_signal_task_attaches_to_builder_and_drops_on_despawn() {
    use jonmo::Builder;

    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));
    let produced = Arc::new(AtomicUsize::new(0));

    // Call `.task()` directly on `Signal<Item = NonClone>` (not mapped to `()`).
    let direct_non_clone_task = signal::from_system({
        let drops = Arc::clone(&drops);
        let produced = Arc::clone(&produced);
        move |In(()), mut step: Local<usize>| {
            *step += 1;
            produced.fetch_add(1, Ordering::SeqCst);
            NonClone::new(format!("task:{}", *step), &drops)
        }
    })
    .task();

    let holder = Builder::new()
        .hold_tasks([direct_non_clone_task])
        .spawn(app.world_mut());
    app.update();
    app.update();
    assert_eq!(produced.load(Ordering::SeqCst), 2);
    assert_eq!(drops.load(Ordering::SeqCst), 2);

    app.world_mut().despawn(holder);
    app.update();
    assert_eq!(produced.load(Ordering::SeqCst), 2);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert_eq!(system_entity_count(app.world_mut()), 0);
}

#[test]
fn input_retaining_combinators_macros_and_dynamic_spec_default_to_move_only_delivery() {
    let mut app = test_app();
    let drops = Arc::new(AtomicUsize::new(0));
    let retained_observed = Arc::new(Mutex::new(Vec::<String>::new()));
    let sum_observed = Arc::new(Mutex::new(Vec::<String>::new()));
    let single_zip_observed = Arc::new(Mutex::new(Vec::<String>::new()));

    // 1. Input-retaining combinators (`always`, `dedupe`, `filter`, `zip`, `cache`, `broadcast`)
    //    and macros (`signal::sum!`, `signal::zip!`) map cleanly into `NonClone` outputs.
    let always_claim = signal::always(7_i32)
        .dedupe()
        .filter(|In(value): In<i32>| value > 0)
        .zip(signal::once(3_i32).cache().broadcast())
        .map_in({
            let drops = Arc::clone(&drops);
            move |(left, right)| NonClone::new(format!("retained:{}", left + right), &drops)
        })
        .map({
            let retained_observed = Arc::clone(&retained_observed);
            move |In(item): In<NonClone>| {
                retained_observed
                    .lock()
                    .expect("retained lock")
                    .push(item.label.clone());
            }
        })
        .register(app.world_mut());

    let sum_claim = signal::sum!(
        signal::once(AddToNonClone {
            value: 10,
            drops: Arc::clone(&drops),
        }),
        signal::once(AddToNonClone {
            value: 32,
            drops: Arc::clone(&drops),
        }),
    )
    .map({
        let sum_observed = Arc::clone(&sum_observed);
        move |In(item): In<NonClone>| {
            sum_observed.lock().expect("sum lock").push(item.label.clone());
        }
    })
    .register(app.world_mut());

    let single_zip_claim = signal::zip!(signal::once(NonClone::new("single_zip", &drops)))
        .map({
            let single_zip_observed = Arc::clone(&single_zip_observed);
            move |In((item,)): In<(NonClone,)>| {
                single_zip_observed
                    .lock()
                    .expect("single_zip lock")
                    .push(item.label.clone());
            }
        })
        .register(app.world_mut());

    app.update();
    assert_eq!(*retained_observed.lock().expect("retained lock"), vec!["retained:10"]);
    assert_eq!(*sum_observed.lock().expect("sum lock"), vec!["sum:42"]);
    assert_eq!(
        *single_zip_observed.lock().expect("single_zip lock"),
        vec!["single_zip"]
    );
    always_claim.release(app.world_mut());
    sum_claim.release(app.world_mut());
    single_zip_claim.release(app.world_mut());

    // 2. `filter` and `zip` use `OutputDelivery::move_only()`:
    //    - `filter` clones once for `FilterSystemAdapter`'s `In<CloneProbe>` predicate and 0 times
    //      for 1-dependent delivery; calling `.broadcast()` on `filter` appends 1 canonical
    //      broadcast entity (3 entities total: source + filter + broadcast).
    let filter_clones = Arc::new(AtomicUsize::new(0));
    let filter_claim = signal::once(CloneProbe::new(4, &filter_clones))
        .filter(|In(probe): In<CloneProbe>| probe.value == 4)
        .map(|In(probe): In<CloneProbe>| {
            assert_eq!(probe.value, 4);
        })
        .register(app.world_mut());
    app.update();
    assert_eq!(
        filter_clones.load(Ordering::SeqCst),
        1,
        "filter must clone only once for predicate evaluation and move output downstream"
    );
    filter_claim.release(app.world_mut());
    assert_eq!(system_entity_count(app.world_mut()), 0);

    let filter_broadcast = signal::once(1_i32).filter(|In(value): In<i32>| value > 0).broadcast();
    let fb_claim = filter_broadcast.register(app.world_mut());
    assert_eq!(
        system_entity_count(app.world_mut()),
        3,
        "non-replayable move_only `Filter` wrapped in `.broadcast()` must materialize source + filter + broadcast"
    );
    fb_claim.release(app.world_mut());
    assert_eq!(system_entity_count(app.world_mut()), 0);

    //    - `zip` clones once per cached side (`left_cache.clone().zip(right_cache.clone())`) and 0
    //      times on 1-dependent output delivery; `.broadcast()` appends 1 canonical broadcast
    //      entity.
    let left_clones = Arc::new(AtomicUsize::new(0));
    let right_clones = Arc::new(AtomicUsize::new(0));
    let zip_claim = signal::once(CloneProbe::new(2, &left_clones))
        .zip(signal::once(CloneProbe::new(5, &right_clones)))
        .map(|In((left, right)): In<(CloneProbe, CloneProbe)>| {
            assert_eq!(left.value + right.value, 7);
        })
        .register(app.world_mut());
    app.update();
    assert_eq!(left_clones.load(Ordering::SeqCst), 1);
    assert_eq!(right_clones.load(Ordering::SeqCst), 1);
    zip_claim.release(app.world_mut());
    assert_eq!(system_entity_count(app.world_mut()), 0);

    // 3. `DynamicSignalOperatorSpec` defaults to `OutputDelivery::move_only()`: attaching two
    //    distinct downstream consumers to a default `DynamicSignalOperatorSpec` stable output
    //    entity fails preflight with `SignalRegistrationError::MoveOnlyFanout`, whereas
    //    `.output_delivery(OutputDelivery::broadcast())` permits fan-out.
    struct DynamicFanoutProbe {
        broadcast_override: bool,
    }

    impl Signal for DynamicFanoutProbe {
        type Item = (i32, i32);

        fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
            let broadcast_override = self.broadcast_override;
            let selected = signal::once(21_i32).cache().broadcast();
            ScalarSignalMaterializer::new(move |registrar| {
                let manager = registrar.signal_entity(
                    (),
                    move |In(()), mut emitted: Local<bool>| {
                        (!core::mem::replace(&mut *emitted, true)).then(|| {
                            DynamicManagerUpdate::new(
                                [ScalarSlotCommand::Select {
                                    signal: selected.clone(),
                                }],
                                DynamicEmissions::<i32>::None,
                            )
                        })
                    },
                    OutputDelivery::move_only(),
                )?;
                let spec = DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<i32>()
                    .adapter(|(), value| DynamicEmissions::One(value * 2))
                    .scalar_output(|values| values.into_iter().last())
                    .same_signal_identity(SameSignalIdentity::KeepCurrent);
                let spec = if broadcast_override {
                    spec.output_delivery(OutputDelivery::broadcast())
                } else {
                    spec
                };
                let dynamic_out = registrar.dynamic_operator(spec)?;
                let left = registrar.signal_entity(
                    dynamic_out,
                    |In(value): In<i32>| Some((Some(value + 1), None::<i32>)),
                    OutputDelivery::move_only(),
                )?;
                let right = registrar.signal_entity(
                    dynamic_out,
                    |In(value): In<i32>| Some((None::<i32>, Some(value + 2))),
                    OutputDelivery::move_only(),
                )?;
                registrar.signal_entity(
                    [left, right],
                    #[allow(clippy::type_complexity)]
                    |In((left_opt, right_opt)): In<(Option<i32>, Option<i32>)>,
                     mut left_cached: Local<Option<i32>>,
                     mut right_cached: Local<Option<i32>>| {
                        if left_opt.is_some() {
                            *left_cached = left_opt;
                        }
                        if right_opt.is_some() {
                            *right_cached = right_opt;
                        }
                        (*left_cached).zip(*right_cached)
                    },
                    OutputDelivery::move_only(),
                )
            })
        }
    }

    let err = DynamicFanoutProbe {
        broadcast_override: false,
    }
    .try_register(app.world_mut())
    .expect_err("default move_only DynamicSignalOperatorSpec must reject multi-dependent fan-out");
    assert_eq!(err, SignalRegistrationError::MoveOnlyFanout);
    assert_eq!(system_entity_count(app.world_mut()), 0);

    let fanout_results = Arc::new(Mutex::new(Vec::<(i32, i32)>::new()));
    let fanout_claim = DynamicFanoutProbe {
        broadcast_override: true,
    }
    .map({
        let fanout_results = Arc::clone(&fanout_results);
        move |In(pair): In<(i32, i32)>| {
            fanout_results.lock().expect("fanout lock").push(pair);
        }
    })
    .try_register(app.world_mut())
    .expect("explicit OutputDelivery::broadcast() on DynamicSignalOperatorSpec must allow fan-out");
    app.update();
    assert_eq!(*fanout_results.lock().expect("fanout lock"), vec![(43, 44)]);
    fanout_claim.release(app.world_mut());
    assert_eq!(system_entity_count(app.world_mut()), 0);
}
