//! Chain must translate and forward each collection batch exactly once.

extern crate alloc;

use alloc::sync::Arc;
use bevy_app::{App, Last};
use bevy_ecs::prelude::*;
#[cfg(feature = "imperative-signal-evaluation")]
use jonmo::graph::{downcast_signal_value, run_signal_subgraph_once};
use jonmo::{OutputDelivery, VecSignalMaterializer, prelude::*};
use std::sync::Mutex;

type Batch = Vec<VecDiff<i32>>;

#[derive(Clone, Default)]
struct Journal(Arc<Mutex<Vec<Batch>>>);

impl Journal {
    fn record(&self, batch: Batch) {
        self.0.lock().expect("journal lock").push(batch);
    }

    fn take(&self) -> Vec<Batch> {
        core::mem::take(&mut *self.0.lock().expect("journal lock"))
    }
}

struct ObserveVec {
    source: VecSignalMaterializer<i32>,
    journal: Journal,
}

impl ObserveVec {
    fn new(source: impl SignalVec<Item = i32>, journal: Journal) -> Self {
        Self {
            source: Box::new(source).into_signal_vec_materializer(),
            journal,
        }
    }
}

impl SignalVec for ObserveVec {
    type Item = i32;

    fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
        let Self { source, journal } = *self;
        VecSignalMaterializer::new(move |registrar| {
            let source = registrar.materialize_signal_vec(source)?;
            registrar.signal_vec_entity(
                source,
                move |In(batch): In<Batch>| {
                    journal.record(batch.clone());
                    Some(batch)
                },
                OutputDelivery::broadcast(),
            )
        })
    }
}

fn assert_batches(actual: &[Batch], expected: &[Batch]) {
    // VecDiff's PartialEq implementation is private to the library's unit tests.
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

fn apply_batches(values: &mut Vec<i32>, batches: Vec<Batch>) {
    for batch in batches {
        for diff in batch {
            diff.apply_to_vec(values);
        }
    }
}

struct Fixture {
    app: App,
    left: MutableVec<i32>,
    right: MutableVec<i32>,
    left_source: SignalClaim,
    right_source: SignalClaim,
    chain: SignalClaim,
    observer: SignalClaim,
    downstream: SignalClaim,
    left_batches: Journal,
    right_batches: Journal,
    chain_batches: Journal,
    downstream_batches: Journal,
    values: Vec<i32>,
    downstream_values: Vec<i32>,
}

impl Fixture {
    fn new(left: &[i32], right: &[i32]) -> Self {
        Self::with_source_order(left, right, false, 0)
    }

    fn with_source_order(left: &[i32], right: &[i32], right_first: bool, left_depth: usize) -> Self {
        let mut app = App::new();
        app.add_plugins(JonmoPlugin::default());
        app.init_resource::<bevy_diagnostic::FrameCount>();
        app.add_systems(bevy_app::First, |mut frame: ResMut<bevy_diagnostic::FrameCount>| {
            frame.0 += 1;
        });
        let (left, right) = if right_first {
            let right = MutableVec::builder().values(right.to_vec()).spawn(app.world_mut());
            let left = MutableVec::builder().values(left.to_vec()).spawn(app.world_mut());
            (left, right)
        } else {
            let left = MutableVec::builder().values(left.to_vec()).spawn(app.world_mut());
            let right = MutableVec::builder().values(right.to_vec()).spawn(app.world_mut());
            (left, right)
        };
        let left_source = left.signal_vec().register(app.world_mut());
        let right_source = right.signal_vec().register(app.world_mut());
        let left_batches = Journal::default();
        let right_batches = Journal::default();
        let chain_batches = Journal::default();
        let downstream_batches = Journal::default();
        let mut left_signal = ObserveVec::new(left.signal_vec(), left_batches.clone()).schedule::<Last>();
        for _ in 0..left_depth {
            left_signal = left_signal.map_in(core::convert::identity).schedule::<Last>();
        }
        let chained = left_signal
            .chain(ObserveVec::new(right.signal_vec(), right_batches.clone()))
            .schedule::<Last>()
            .broadcast();
        let chain = chained.clone().register(app.world_mut());
        let observer = chained
            .clone()
            .for_each({
                let journal = chain_batches.clone();
                move |In(batch): In<Batch>| journal.record(batch)
            })
            .register(app.world_mut());
        let downstream = ObserveVec::new(chained, downstream_batches.clone())
            .map_in(core::convert::identity)
            .register(app.world_mut());
        let mut fixture = Self {
            app,
            left,
            right,
            left_source,
            right_source,
            chain,
            observer,
            downstream,
            left_batches,
            right_batches,
            chain_batches,
            downstream_batches,
            values: Vec::new(),
            downstream_values: Vec::new(),
        };
        fixture.app.update();
        apply_batches(&mut fixture.values, fixture.chain_batches.take());
        apply_batches(&mut fixture.downstream_values, fixture.downstream_batches.take());
        fixture.assert_values();
        fixture.left_batches.take();
        fixture.right_batches.take();
        fixture
    }

    #[cfg(feature = "imperative-signal-evaluation")]
    fn poll_source(&mut self, left: bool, expected: Batch) {
        let source = if left { &self.left_source } else { &self.right_source };
        let actual = run_signal_subgraph_once(self.app.world_mut(), source.entity())
            .and_then(downcast_signal_value::<Batch>)
            .expect("source evaluation must produce pending mutations");
        println!("source left={left}: {actual:?}");
        assert_batches(&[actual], &[expected]);
    }

    #[cfg(feature = "imperative-signal-evaluation")]
    fn push_left(&mut self, value: i32) {
        self.left.write(self.app.world_mut()).push(value);
        self.poll_source(true, vec![VecDiff::Push { value }]);
    }

    #[cfg(feature = "imperative-signal-evaluation")]
    fn poll_chain(&mut self) -> Option<Batch> {
        let output = run_signal_subgraph_once(self.app.world_mut(), self.chain.entity())
            .and_then(downcast_signal_value::<Batch>);
        println!("chain evaluation: {output:?}");
        output
    }

    fn check(&mut self, expected: Vec<Batch>) {
        let actual = self.chain_batches.take();
        let downstream = self.downstream_batches.take();
        println!("chain batches: {actual:?}; downstream: {downstream:?}; expected: {expected:?}");
        assert_batches(&actual, &expected);
        assert_batches(&downstream, &expected);
        apply_batches(&mut self.values, actual);
        apply_batches(&mut self.downstream_values, downstream);
        self.assert_values();
    }

    fn assert_values(&self) {
        let expected = self
            .left
            .read(self.app.world())
            .iter()
            .copied()
            .chain(self.right.read(self.app.world()).iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(self.values, expected);
        assert_eq!(self.downstream_values, expected);
    }

    fn release(self) {
        let mut app = self.app;
        self.downstream.release(app.world_mut());
        self.observer.release(app.world_mut());
        self.chain.release(app.world_mut());
        self.left_source.release(app.world_mut());
        self.right_source.release(app.world_mut());
    }
}

#[cfg(feature = "imperative-signal-evaluation")]
fn two_left_polls(downstream_poll: bool) {
    let mut fixture = Fixture::new(&[], &[]);
    fixture.push_left(1);
    fixture.push_left(2);
    let expected = vec![vec![VecDiff::Push { value: 1 }], vec![VecDiff::Push { value: 2 }]];
    if downstream_poll {
        let actual = run_signal_subgraph_once(fixture.app.world_mut(), fixture.downstream.entity())
            .and_then(downcast_signal_value::<Batch>)
            .expect("downstream evaluation output");
        assert_batches(&[actual], &[expected.iter().flatten().cloned().collect()]);
    }
    fixture.app.update();
    assert_batches(&fixture.left_batches.take(), &expected);
    fixture.check(expected);
    fixture.release();
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn separate_left_source_polls_preserve_pushes_one_and_two() {
    two_left_polls(false);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn downstream_level_poll_preserves_each_chain_batch() {
    two_left_polls(true);
}

#[test]
fn simultaneous_non_idempotent_left_and_right_diffs() {
    let mut fixture = Fixture::new(&[10, 11], &[20, 21]);
    fixture.left.write(fixture.app.world_mut()).pop();
    fixture.right.write(fixture.app.world_mut()).insert(1, 22);
    fixture.app.update();
    assert_batches(&fixture.left_batches.take(), &[vec![VecDiff::Pop]]);
    assert_batches(
        &fixture.right_batches.take(),
        &[vec![VecDiff::InsertAt { index: 1, value: 22 }]],
    );
    // Equal-level producers may execute in either order; offsets must match that order.
    let left_first = fixture
        .chain_batches
        .0
        .lock()
        .expect("journal lock")
        .first()
        .is_some_and(|batch| matches!(batch.as_slice(), [VecDiff::RemoveAt { index: 1 }]));
    fixture.check(if left_first {
        vec![
            vec![VecDiff::RemoveAt { index: 1 }],
            vec![VecDiff::InsertAt { index: 2, value: 22 }],
        ]
    } else {
        vec![
            vec![VecDiff::InsertAt { index: 3, value: 22 }],
            vec![VecDiff::RemoveAt { index: 1 }],
        ]
    });
    fixture.release();
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn repeated_chain_polls_in_one_frame_do_not_replay_prior_output() {
    let mut fixture = Fixture::new(&[], &[]);
    fixture.app.update();
    fixture.push_left(1);
    assert_batches(
        &[fixture.poll_chain().expect("first push")],
        &[vec![VecDiff::Push { value: 1 }]],
    );
    assert!(fixture.poll_chain().is_none(), "idle poll must not repeat output");
    fixture.push_left(2);
    assert_batches(
        &[fixture.poll_chain().expect("second push")],
        &[vec![VecDiff::Push { value: 2 }]],
    );
    assert!(fixture.poll_chain().is_none());
    fixture.app.update();
    fixture.check(vec![vec![VecDiff::Push { value: 1 }], vec![VecDiff::Push { value: 2 }]]);
    fixture.app.update();
    fixture.check(vec![]);
    fixture.release();
}

#[cfg(feature = "imperative-signal-evaluation")]
fn replace_then_suffix(left: bool) {
    let mut fixture = Fixture::new(&[10, 11], &[20, 21]);
    let values = if left { &fixture.left } else { &fixture.right };
    values.write(fixture.app.world_mut()).replace(vec![3]);
    fixture.poll_source(left, vec![VecDiff::Replace { values: vec![3] }]);
    let values = if left { &fixture.left } else { &fixture.right };
    values.write(fixture.app.world_mut()).push(4);
    fixture.poll_source(left, vec![VecDiff::Push { value: 4 }]);
    fixture.app.update();
    let observed = if left {
        fixture.left_batches.take()
    } else {
        fixture.right_batches.take()
    };
    assert_batches(
        &observed,
        &[
            vec![VecDiff::Replace { values: vec![3] }],
            vec![VecDiff::Push { value: 4 }],
        ],
    );
    let expected = if left {
        vec![
            vec![
                VecDiff::RemoveAt { index: 1 },
                VecDiff::RemoveAt { index: 0 },
                VecDiff::InsertAt { index: 0, value: 3 },
            ],
            vec![VecDiff::InsertAt { index: 1, value: 4 }],
        ]
    } else {
        vec![
            vec![VecDiff::Pop, VecDiff::Pop, VecDiff::Push { value: 3 }],
            vec![VecDiff::Push { value: 4 }],
        ]
    };
    fixture.check(expected);
    fixture.release();
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn left_replace_then_separately_polled_suffix_preserves_right() {
    replace_then_suffix(true);
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn right_replace_then_separately_polled_suffix_preserves_left() {
    replace_then_suffix(false);
}

#[test]
fn deeper_left_source_uses_right_producer_order_not_left_first_assumption() {
    let mut fixture = Fixture::with_source_order(&[10, 11], &[20, 21], true, 2);
    fixture.left.write(fixture.app.world_mut()).pop();
    fixture.right.write(fixture.app.world_mut()).insert(1, 22);
    fixture.app.update();
    fixture.check(vec![
        vec![VecDiff::InsertAt { index: 3, value: 22 }],
        vec![VecDiff::RemoveAt { index: 1 }],
    ]);
    fixture.release();
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn ordinary_scalar_after_chain_still_delivers_only_last_output_per_drain() {
    let mut fixture = Fixture::new(&[], &[]);
    let callbacks = Arc::new(Mutex::new(Vec::new()));
    let downstream = Arc::new(Mutex::new(Vec::new()));
    let scalar = fixture
        .left
        .signal_vec()
        .chain(fixture.right.signal_vec())
        .for_each({
            let callbacks = callbacks.clone();
            move |In(batch): In<Batch>| {
                let value = batch.into_iter().find_map(|diff| match diff {
                    VecDiff::Push { value } => Some(value),
                    _ => None,
                });
                if let Some(value) = value {
                    callbacks.lock().expect("callback lock").push(value);
                }
                value
            }
        })
        .map({
            let downstream = downstream.clone();
            move |In(value): In<i32>| downstream.lock().expect("downstream lock").push(value)
        })
        .register(fixture.app.world_mut());
    fixture.app.update();
    fixture.push_left(1);
    fixture.push_left(2);
    fixture.app.update();
    assert_eq!(*callbacks.lock().expect("callback lock"), [1, 2]);
    assert_eq!(*downstream.lock().expect("downstream lock"), [2]);
    fixture.check(vec![vec![VecDiff::Push { value: 1 }], vec![VecDiff::Push { value: 2 }]]);
    scalar.release(fixture.app.world_mut());
    fixture.release();
}

#[test]
fn established_diff_translations_on_both_sides() {
    use VecDiff::*;

    for left in [true, false] {
        let mut fixture = Fixture::new(&[1, 2, 3], &[10, 20, 30]);
        let offset = if left { 0 } else { 3 };
        let cases = vec![
            (
                InsertAt { index: 1, value: 4 },
                vec![InsertAt {
                    index: offset + 1,
                    value: 4,
                }],
            ),
            (
                UpdateAt { index: 1, value: 5 },
                vec![UpdateAt {
                    index: offset + 1,
                    value: 5,
                }],
            ),
            (
                Move {
                    old_index: 3,
                    new_index: 0,
                },
                vec![Move {
                    old_index: offset + 3,
                    new_index: offset,
                }],
            ),
            (RemoveAt { index: 1 }, vec![RemoveAt { index: offset + 1 }]),
            (Pop, if left { vec![RemoveAt { index: 2 }] } else { vec![Pop] }),
            (
                Push { value: 6 },
                if left {
                    vec![InsertAt { index: 2, value: 6 }]
                } else {
                    vec![Push { value: 6 }]
                },
            ),
            (
                Replace { values: vec![7, 8] },
                if left {
                    vec![
                        RemoveAt { index: 2 },
                        RemoveAt { index: 1 },
                        RemoveAt { index: 0 },
                        InsertAt { index: 0, value: 7 },
                        InsertAt { index: 1, value: 8 },
                    ]
                } else {
                    vec![Pop, Pop, Pop, Push { value: 7 }, Push { value: 8 }]
                },
            ),
            (
                Clear,
                if left {
                    vec![RemoveAt { index: 1 }, RemoveAt { index: 0 }]
                } else {
                    vec![Pop, Pop]
                },
            ),
            (
                Push { value: 9 },
                if left {
                    vec![InsertAt { index: 0, value: 9 }]
                } else {
                    vec![Push { value: 9 }]
                },
            ),
        ];
        for (diff, expected) in cases {
            let source = if left { &fixture.left } else { &fixture.right };
            {
                let mut values = source.write(fixture.app.world_mut());
                match diff {
                    Replace { values: replacement } => values.replace(replacement),
                    InsertAt { index, value } => values.insert(index, value),
                    UpdateAt { index, value } => values.set(index, value),
                    Move { old_index, new_index } => values.move_item(old_index, new_index),
                    RemoveAt { index } => {
                        values.remove(index);
                    }
                    Push { value } => values.push(value),
                    Pop => {
                        values.pop();
                    }
                    Clear => values.clear(),
                }
            }
            fixture.app.update();
            fixture.check(vec![expected]);
        }
        fixture.release();
    }
}

#[test]
fn empty_other_side_keeps_replace_clear_push_and_pop_semantics() {
    use VecDiff::*;

    for left in [true, false] {
        let mut fixture = Fixture::new(&[], &[]);
        let source = if left {
            fixture.left.clone()
        } else {
            fixture.right.clone()
        };
        source.write(fixture.app.world_mut()).replace(vec![1, 2]);
        fixture.app.update();
        fixture.check(vec![vec![Replace { values: vec![1, 2] }]]);
        source.write(fixture.app.world_mut()).pop();
        fixture.app.update();
        fixture.check(vec![vec![Pop]]);
        source.write(fixture.app.world_mut()).push(3);
        fixture.app.update();
        fixture.check(vec![vec![Push { value: 3 }]]);
        source.write(fixture.app.world_mut()).clear();
        fixture.app.update();
        fixture.check(vec![vec![Clear]]);
        source.write(fixture.app.world_mut()).replace(vec![]);
        fixture.app.update();
        fixture.check(vec![vec![Replace { values: vec![] }]]);
        fixture.release();
    }
}

#[cfg(feature = "imperative-signal-evaluation")]
#[test]
fn independent_chain_materializations_do_not_share_lengths() {
    let mut fixture = Fixture::new(&[10], &[20]);
    let journal = Journal::default();
    let second = fixture
        .left
        .signal_vec()
        .chain(fixture.right.signal_vec())
        .for_each({
            let journal = journal.clone();
            move |In(batch): In<Batch>| journal.record(batch)
        })
        .register(fixture.app.world_mut());
    fixture.app.update();
    let mut second_values = Vec::new();
    apply_batches(&mut second_values, journal.take());
    assert_eq!(second_values, [10, 20]);
    fixture.push_left(1);
    fixture.push_left(2);
    fixture.app.update();
    let expected = vec![
        vec![VecDiff::InsertAt { index: 1, value: 1 }],
        vec![VecDiff::InsertAt { index: 2, value: 2 }],
    ];
    fixture.check(expected.clone());
    let actual = journal.take();
    assert_batches(&actual, &expected);
    apply_batches(&mut second_values, actual);
    assert_eq!(second_values, [10, 1, 2, 20]);
    second.release(fixture.app.world_mut());
    fixture.release();
}

mod canceled_chain_delivery {
    use super::*;
    use bevy_ecs::schedule::ScheduleLabel;

    #[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
    struct Producer;

    #[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
    struct Consumer;

    fn run_case(established: bool, cancel: bool) {
        let mut app = App::new();
        app.add_plugins(JonmoPlugin::new::<Producer>().with_schedule::<Consumer>());
        app.init_resource::<bevy_diagnostic::FrameCount>();
        app.update();
        let left = MutableVec::builder().values([10]).spawn(app.world_mut());
        let right = MutableVec::builder().values([20]).spawn(app.world_mut());
        let raw = Journal::default();
        let delivered = Journal::default();
        let chained = left
            .signal_vec()
            .schedule::<Producer>()
            .chain(right.signal_vec().schedule::<Producer>())
            .schedule::<Consumer>();
        // Consume the cache into an ordinary retained terminal: retaining the cache itself
        // would keep replay maintenance active and make this release non-final upstream.
        let terminal = ObserveVec::new(chained, raw.clone())
            .cache()
            .for_each({
                let delivered = delivered.clone();
                move |In(batch): In<Batch>| delivered.record(batch)
            })
            .schedule::<Consumer>()
            .broadcast();
        let claim = terminal
            .clone()
            .try_register(app.world_mut())
            .expect("initial registration");
        let identity = claim.entity();
        if established {
            app.world_mut().run_schedule(Producer);
            app.world_mut().run_schedule(Consumer);
            let initial = delivered.take();
            let mut values = Vec::new();
            apply_batches(&mut values, initial.clone());
            println!("established initial: raw={:?}, delivered={initial:?}", raw.take());
            assert_eq!(values, [10, 20], "setup must establish the authoritative cache first");
            left.write(app.world_mut()).push(11);
        }
        app.world_mut().run_schedule(Producer);
        assert!(
            raw.take().is_empty(),
            "Producer must not execute the chain's Consumer end"
        );
        assert!(
            delivered.take().is_empty(),
            "no Consumer delivery before its schedule runs"
        );
        println!("established={established}, cancel={cancel}: Producer ran; Consumer has not run");
        let claim = if cancel {
            claim.try_release(app.world_mut()).expect("final terminal release");
            assert!(
                app.world().get_entity(identity.entity()).is_ok(),
                "ordinary identity remains retained"
            );
            #[cfg(feature = "imperative-signal-evaluation")]
            assert!(
                run_signal_subgraph_once(app.world_mut(), identity).is_none(),
                "released terminal is inactive"
            );
            app.world_mut().run_schedule(Consumer);
            app.world_mut().run_schedule(Producer);
            app.world_mut().run_schedule(Consumer);
            assert!(
                raw.take().is_empty(),
                "no hidden cache maintenance may consume canceled chain output"
            );
            assert!(
                delivered.take().is_empty(),
                "released terminal must not receive pending delivery"
            );
            let claim = terminal
                .clone()
                .try_register(app.world_mut())
                .expect("reactivate retained terminal");
            assert_eq!(
                claim.entity(),
                identity,
                "reactivation must reuse the retained ordinary terminal"
            );
            claim
        } else {
            claim
        };
        for _ in 0..3 {
            app.world_mut().resource_mut::<bevy_diagnostic::FrameCount>().0 += 1;
            app.world_mut().run_schedule(Producer);
            app.world_mut().run_schedule(Consumer);
        }
        let raw = raw.take();
        let delivered = delivered.take();
        println!("established={established}, cancel={cancel}: raw={raw:?}, delivered={delivered:?}");
        let mut values = if established && !cancel {
            vec![10, 20]
        } else {
            Vec::new()
        };
        apply_batches(&mut values, delivered);
        let expected = if established { vec![10, 11, 20] } else { vec![10, 20] };
        claim.try_release(app.world_mut()).expect("release final test claim");
        assert_eq!(
            values, expected,
            "reactivated cache must deliver the current concatenated vector"
        );
    }

    #[test]
    fn pending_bootstrap_without_release_control() {
        run_case(false, false);
    }

    #[test]
    fn pending_bootstrap_final_release_then_reactivation_must_replay_current() {
        run_case(false, true);
    }

    #[test]
    fn established_then_queued_update_without_release_control() {
        run_case(true, false);
    }

    #[test]
    fn established_then_queued_update_final_release_must_replay_current() {
        run_case(true, true);
    }
}

mod transformed_chain_delivery {
    use super::*;
    use bevy_ecs::schedule::ScheduleLabel;

    #[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
    struct Producer;

    #[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
    struct ChainConsumer;

    #[derive(Default, Debug, Clone, PartialEq, Eq, Hash, ScheduleLabel)]
    struct CacheConsumer;

    struct PhaseFixture {
        app: App,
        left: MutableVec<i32>,
        right: MutableVec<i32>,
        terminal: Broadcast<()>,
        left_raw: Journal,
        right_raw: Journal,
        transformed: Journal,
        delivered: Journal,
    }

    impl PhaseFixture {
        fn new(shared_source: bool) -> Self {
            let mut app = App::new();
            app.add_plugins(
                JonmoPlugin::new::<Producer>()
                    .with_schedule::<ChainConsumer>()
                    .with_schedule::<CacheConsumer>(),
            );
            app.init_resource::<bevy_diagnostic::FrameCount>();
            app.update();
            let left = MutableVec::builder().values([1, 2]).spawn(app.world_mut());
            let right = if shared_source {
                left.clone()
            } else {
                MutableVec::builder().values([3, 4]).spawn(app.world_mut())
            };
            let left_raw = Journal::default();
            let right_raw = Journal::default();
            let transformed = Journal::default();
            let delivered = Journal::default();

            // Schedule the canonical sources before allocating distinct role adapters. Scheduling
            // two aliases to different roles would overwrite the shared source's schedule.
            let left_source = left.signal_vec().schedule::<Producer>().broadcast();
            let right_source = right.signal_vec().schedule::<Producer>().broadcast();
            let left_claim = left_source.clone().register(app.world_mut());
            let right_claim = right_source.clone().register(app.world_mut());
            assert_eq!(
                left_claim.entity() == right_claim.entity(),
                shared_source,
                "source identity must match the requested role topology"
            );
            left_claim.release(app.world_mut());
            right_claim.release(app.world_mut());

            let left_role = ObserveVec::new(left_source, left_raw.clone())
                .schedule::<ChainConsumer>()
                .map_in(|value| value + 100);
            let right_role = ObserveVec::new(right_source, right_raw.clone())
                .schedule::<ChainConsumer>()
                .map_in(|value| value + 200);
            let chained = left_role.chain(right_role).schedule::<ChainConsumer>();
            // Consume every intermediate handle. Only the ordinary terminal is retained, so no
            // direct cache retainer can keep upstream replay maintenance alive after final release.
            let terminal = ObserveVec::new(chained, transformed.clone())
                .schedule::<ChainConsumer>()
                .cache()
                .schedule::<CacheConsumer>()
                .for_each({
                    let delivered = delivered.clone();
                    move |In(batch): In<Batch>| delivered.record(batch)
                })
                .schedule::<CacheConsumer>()
                .broadcast();
            Self {
                app,
                left,
                right,
                terminal,
                left_raw,
                right_raw,
                transformed,
                delivered,
            }
        }

        fn expected(&self) -> Vec<i32> {
            self.left
                .read(self.app.world())
                .iter()
                .map(|value| value + 100)
                .chain(self.right.read(self.app.world()).iter().map(|value| value + 200))
                .collect()
        }

        fn producer(&mut self) {
            self.app.world_mut().resource_mut::<bevy_diagnostic::FrameCount>().0 += 1;
            self.app.world_mut().run_schedule(Producer);
            assert!(
                self.left_raw.take().is_empty(),
                "Producer must not consume the left role"
            );
            assert!(
                self.right_raw.take().is_empty(),
                "Producer must not consume the right role"
            );
            assert!(
                self.transformed.take().is_empty(),
                "Producer must not execute chain output"
            );
            assert!(
                self.delivered.take().is_empty(),
                "Producer must not execute cache delivery"
            );
        }

        fn chain(&mut self, expected_left: Vec<Batch>, expected_right: Vec<Batch>, values: &mut Vec<i32>) {
            self.app.world_mut().run_schedule(ChainConsumer);
            assert_batches(&self.left_raw.take(), &expected_left);
            assert_batches(&self.right_raw.take(), &expected_right);
            let transformed = self.transformed.take();
            println!("chain consumed: transformed={transformed:?}");
            assert_eq!(
                transformed.len(),
                expected_left.len() + expected_right.len(),
                "each nonempty source-role batch must reach the final chain exactly once"
            );
            apply_batches(values, transformed);
            assert_eq!(
                *values,
                self.expected(),
                "chain must transform all raw batches before the gap"
            );
            assert!(
                self.delivered.take().is_empty(),
                "ChainConsumer must not run the explicitly separate CacheConsumer"
            );
        }

        fn cache(&mut self, values: &mut Vec<i32>) {
            self.app.world_mut().run_schedule(CacheConsumer);
            let delivered = self.delivered.take();
            println!("cache consumed: delivered={delivered:?}");
            apply_batches(values, delivered);
            assert_eq!(
                *values,
                self.expected(),
                "unreleased cache must consume complete chain output"
            );
        }
    }

    fn run_gap(established: bool, cancel: bool, shared_source: bool) {
        let mut fixture = PhaseFixture::new(shared_source);
        let claim = fixture.terminal.clone().register(fixture.app.world_mut());
        let identity = claim.entity();
        let mut chain_values = Vec::new();
        let mut cache_values = Vec::new();
        fixture.producer();
        fixture.chain(
            vec![vec![VecDiff::Replace { values: vec![1, 2] }]],
            vec![vec![VecDiff::Replace {
                values: if shared_source { vec![1, 2] } else { vec![3, 4] },
            }]],
            &mut chain_values,
        );
        if established {
            fixture.cache(&mut cache_values);
            fixture.left.write(fixture.app.world_mut()).push(5);
            fixture.producer();
            fixture.chain(
                vec![vec![VecDiff::Push { value: 5 }]],
                if shared_source {
                    vec![vec![VecDiff::Push { value: 5 }]]
                } else {
                    vec![]
                },
                &mut chain_values,
            );
        }
        println!(
            "gap established={established}, cancel={cancel}, shared_source={shared_source}: chain={chain_values:?}, cache last delivered={cache_values:?}; CacheConsumer has NOT run for pending output"
        );
        let claim = if cancel {
            claim
                .try_release(fixture.app.world_mut())
                .expect("final ordinary terminal release");
            assert!(
                fixture.app.world().get_entity(identity.entity()).is_ok(),
                "ordinary terminal identity must remain retained"
            );
            #[cfg(feature = "imperative-signal-evaluation")]
            assert!(
                run_signal_subgraph_once(fixture.app.world_mut(), identity).is_none(),
                "released terminal is inactive"
            );
            fixture.app.world_mut().run_schedule(CacheConsumer);
            fixture.producer();
            fixture.app.world_mut().run_schedule(ChainConsumer);
            fixture.app.world_mut().run_schedule(CacheConsumer);
            assert!(fixture.left_raw.take().is_empty(), "no retained left role maintenance");
            assert!(
                fixture.right_raw.take().is_empty(),
                "no retained right role maintenance"
            );
            assert!(fixture.transformed.take().is_empty(), "no retained chain maintenance");
            assert!(
                fixture.delivered.take().is_empty(),
                "final release cancels queued cache delivery"
            );
            cache_values.clear();
            let claim = fixture
                .terminal
                .clone()
                .try_register(fixture.app.world_mut())
                .expect("reactivate ordinary terminal");
            assert_eq!(
                claim.entity(),
                identity,
                "reactivation reuses the retained terminal identity"
            );
            claim
        } else {
            claim
        };
        for _ in 0..3 {
            fixture.app.world_mut().resource_mut::<bevy_diagnostic::FrameCount>().0 += 1;
            fixture.app.world_mut().run_schedule(Producer);
            fixture.app.world_mut().run_schedule(ChainConsumer);
            fixture.app.world_mut().run_schedule(CacheConsumer);
        }
        let transformed = fixture.transformed.take();
        let delivered = fixture.delivered.take();
        println!(
            "gap result established={established}, cancel={cancel}, shared_source={shared_source}: transformed={transformed:?}, delivered={delivered:?}"
        );
        apply_batches(&mut cache_values, delivered);
        let expected = fixture.expected();
        claim.release(fixture.app.world_mut());
        assert_eq!(
            cache_values, expected,
            "cache must eventually deliver the current role-tagged concatenation after the transformed-output gap"
        );
    }

    #[test]
    fn warming_transformed_output_without_release_control() {
        run_gap(false, false, false);
    }

    #[test]
    fn warming_transformed_output_final_release_then_reactivation() {
        run_gap(false, true, false);
    }

    #[test]
    fn established_transformed_update_without_release_control() {
        run_gap(true, false, false);
    }

    #[test]
    fn established_transformed_update_final_release_then_reactivation() {
        run_gap(true, true, false);
    }

    #[test]
    fn shared_source_warming_transformed_output_without_release_control() {
        run_gap(false, false, true);
    }

    #[test]
    fn shared_source_warming_transformed_output_final_release_then_reactivation() {
        run_gap(false, true, true);
    }

    #[test]
    fn shared_source_established_transformed_update_without_release_control() {
        run_gap(true, false, true);
    }

    #[test]
    fn shared_source_established_transformed_update_final_release_then_reactivation() {
        run_gap(true, true, true);
    }

    #[test]
    fn shared_source_distinct_roles_preserve_initial_and_separately_polled_batches() {
        let mut fixture = PhaseFixture::new(true);
        let claim = fixture.terminal.clone().register(fixture.app.world_mut());
        let mut chain_values = Vec::new();
        let mut cache_values = Vec::new();
        fixture.producer();
        let initial = vec![vec![VecDiff::Replace { values: vec![1, 2] }]];
        fixture.chain(initial.clone(), initial, &mut chain_values);
        fixture.cache(&mut cache_values);
        assert_eq!(cache_values, [101, 102, 201, 202]);

        let batches = vec![
            vec![VecDiff::Push { value: 3 }],
            vec![
                VecDiff::InsertAt { index: 1, value: 4 },
                VecDiff::UpdateAt { index: 0, value: 5 },
            ],
            vec![VecDiff::Move {
                old_index: 3,
                new_index: 0,
            }],
            vec![VecDiff::Replace { values: vec![7, 8] }],
            vec![VecDiff::Pop, VecDiff::Push { value: 9 }],
        ];
        for batch in &batches {
            {
                let mut values = fixture.left.write(fixture.app.world_mut());
                for diff in batch {
                    match diff {
                        VecDiff::Push { value } => values.push(*value),
                        VecDiff::InsertAt { index, value } => values.insert(*index, *value),
                        VecDiff::UpdateAt { index, value } => values.set(*index, *value),
                        VecDiff::Move { old_index, new_index } => values.move_item(*old_index, *new_index),
                        VecDiff::Replace { values: replacement } => values.replace(replacement.clone()),
                        VecDiff::Pop => {
                            values.pop();
                        }
                        _ => panic!("unexpected edit in shared-source test"),
                    }
                }
            }
            fixture.producer();
        }
        fixture.producer();
        fixture.chain(batches.clone(), batches, &mut chain_values);
        fixture.cache(&mut cache_values);
        assert_eq!(cache_values, [107, 109, 207, 209]);
        claim.release(fixture.app.world_mut());
    }

    mod epoch_repair {
        use super::*;
        use VecDiff::{Clear, InsertAt, Pop, Push, Replace, UpdateAt};
        use alloc::collections::VecDeque;

        type PendingBatches = Arc<Mutex<VecDeque<Batch>>>;

        struct ScriptedVec {
            pending: PendingBatches,
            emitted: Journal,
        }

        impl SignalVec for ScriptedVec {
            type Item = i32;

            fn into_signal_vec_materializer(self: Box<Self>) -> VecSignalMaterializer<i32> {
                let Self { pending, emitted } = *self;
                VecSignalMaterializer::new(move |registrar| {
                    // Ordinary sources let the test choose exactly which batch follows
                    // reactivation, including an incremental batch or one whose chain translation
                    // returns None.
                    let source = registrar.signal_vec_entity(
                        (),
                        move |In(()): In<()>| {
                            let batch = pending.lock().expect("pending batches lock").pop_front();
                            if let Some(batch) = &batch {
                                emitted.record(batch.clone());
                            }
                            batch
                        },
                        OutputDelivery::broadcast(),
                    )?;
                    registrar.apply_schedule(source, Producer)
                })
            }
        }

        struct RepairFixture {
            app: App,
            left: PendingBatches,
            right: PendingBatches,
            emitted: Journal,
            chain: BroadcastVec<i32>,
            terminal: Broadcast<()>,
            transformed: Journal,
            delivered: Journal,
        }

        impl RepairFixture {
            fn new() -> Self {
                let mut app = App::new();
                app.add_plugins(
                    JonmoPlugin::new::<Producer>()
                        .with_schedule::<ChainConsumer>()
                        .with_schedule::<CacheConsumer>(),
                );
                app.init_resource::<bevy_diagnostic::FrameCount>();
                app.update();
                let left = PendingBatches::default();
                let right = PendingBatches::default();
                let emitted = Journal::default();
                let transformed = Journal::default();
                let delivered = Journal::default();
                let chained = ScriptedVec {
                    pending: left.clone(),
                    emitted: emitted.clone(),
                }
                .chain(ScriptedVec {
                    pending: right.clone(),
                    emitted: emitted.clone(),
                })
                .schedule::<ChainConsumer>();
                let chain = ObserveVec::new(chained, transformed.clone())
                    .schedule::<ChainConsumer>()
                    .broadcast();
                let terminal = chain
                    .clone()
                    .cache()
                    .schedule::<CacheConsumer>()
                    .for_each({
                        let delivered = delivered.clone();
                        move |In(batch): In<Batch>| delivered.record(batch)
                    })
                    .schedule::<CacheConsumer>()
                    .broadcast();
                Self {
                    app,
                    left,
                    right,
                    emitted,
                    chain,
                    terminal,
                    transformed,
                    delivered,
                }
            }

            fn register(&mut self) -> SignalClaim {
                self.terminal.clone().register(self.app.world_mut())
            }

            fn emit(&mut self, left: bool, batch: Batch) {
                let pending = if left { &self.left } else { &self.right };
                pending.lock().expect("pending batches lock").push_back(batch.clone());
                self.app.world_mut().resource_mut::<bevy_diagnostic::FrameCount>().0 += 1;
                self.app.world_mut().run_schedule(Producer);
                assert_batches(&self.emitted.take(), &[batch]);
                assert!(
                    self.transformed.take().is_empty(),
                    "Producer must not execute ChainConsumer"
                );
            }

            fn drain_chain(&mut self) -> Vec<Batch> {
                self.app.world_mut().run_schedule(ChainConsumer);
                let batches = self.transformed.take();
                println!("published chain batches: {batches:?}");
                batches
            }

            fn send(&mut self, left: bool, batch: Batch, expected: Vec<Batch>) {
                self.emit(left, batch);
                assert_batches(&self.drain_chain(), &expected);
            }

            fn consume_cache(&mut self, values: &mut Vec<i32>) {
                self.app.world_mut().run_schedule(CacheConsumer);
                let batches = self.delivered.take();
                println!("cache batches (stale initial replay permitted): {batches:?}");
                apply_batches(values, batches);
            }

            fn reactivate(&mut self, claim: SignalClaim) -> SignalClaim {
                let identity = claim.entity();
                claim.release(self.app.world_mut());
                #[cfg(feature = "imperative-signal-evaluation")]
                assert!(run_signal_subgraph_once(self.app.world_mut(), identity).is_none());
                assert!(
                    self.drain_chain().is_empty(),
                    "deactivation alone must not publish output"
                );
                self.app.world_mut().run_schedule(CacheConsumer);
                assert!(
                    self.delivered.take().is_empty(),
                    "inactive terminal must not receive queued output"
                );
                let claim = self.register();
                assert_eq!(claim.entity(), identity);
                claim
            }
        }

        #[test]
        fn repeated_epochs_prefix_last_published_state_preserving_order_and_transients() {
            let mut fixture = RepairFixture::new();
            let mut claim = fixture.register();
            fixture.send(
                true,
                vec![Replace { values: vec![10] }],
                vec![vec![Replace { values: vec![10] }]],
            );
            fixture.send(
                false,
                vec![Replace { values: vec![20] }],
                vec![vec![Push { value: 20 }]],
            );
            let mut published = vec![10, 20];
            let mut cache_values = Vec::new();
            fixture.consume_cache(&mut cache_values);
            assert_eq!(cache_values, published);

            for (left_length, value) in [(1, 21), (2, 22)] {
                fixture.send(false, vec![Push { value }], vec![vec![Push { value }]]);
                published.push(value);
                let before_repair = published.clone();
                assert_ne!(
                    cache_values, before_repair,
                    "cache must lag the last published chain state"
                );
                claim = fixture.reactivate(claim);
                cache_values.clear();
                // A ready cache may replay its old state before the chain executes. Do not require
                // the first callback to be current or the two source sides to bootstrap atomically.
                fixture.consume_cache(&mut cache_values);
                assert!(
                    fixture.drain_chain().is_empty(),
                    "activation without input emits no repair"
                );
                fixture.emit(
                    true,
                    vec![UpdateAt { index: 0, value: 99 }, UpdateAt { index: 0, value }],
                );
                fixture.emit(false, vec![Push { value: value + 10 }]);
                fixture.emit(true, vec![Push { value: 11 }]);
                let ordinary = vec![
                    vec![UpdateAt { index: 0, value: 99 }, UpdateAt { index: 0, value }],
                    vec![Push { value: value + 10 }],
                    vec![InsertAt {
                        index: left_length,
                        value: 11,
                    }],
                ];
                let expected = vec![
                    vec![
                        Replace { values: before_repair },
                        UpdateAt { index: 0, value: 99 },
                        UpdateAt { index: 0, value },
                    ],
                    vec![Push { value: value + 10 }],
                    vec![InsertAt {
                        index: left_length,
                        value: 11,
                    }],
                ];
                assert_batches(&fixture.drain_chain(), &expected);
                apply_batches(&mut published, ordinary);
                for _ in 0..3 {
                    fixture.consume_cache(&mut cache_values);
                }
                assert_eq!(
                    cache_values, published,
                    "stale-ready replay must eventually be repaired"
                );
            }
            claim.release(fixture.app.world_mut());
        }

        #[test]
        fn partial_bootstrap_repairs_only_the_previously_published_side() {
            let mut fixture = RepairFixture::new();
            let claim = fixture.register();
            fixture.send(
                true,
                vec![Replace { values: vec![10] }],
                vec![vec![Replace { values: vec![10] }]],
            );
            assert!(
                fixture.delivered.take().is_empty(),
                "cache has not consumed the first side"
            );
            let claim = fixture.reactivate(claim);
            assert!(fixture.drain_chain().is_empty());
            fixture.send(
                false,
                vec![Replace { values: vec![20] }],
                vec![vec![Replace { values: vec![10] }, Push { value: 20 }]],
            );
            let mut values = Vec::new();
            for _ in 0..3 {
                fixture.consume_cache(&mut values);
            }
            assert_eq!(values, [10, 20]);
            claim.release(fixture.app.world_mut());
        }

        #[test]
        fn pending_epoch_repair_survives_none_translation_and_prefixes_only_next_output() {
            let mut fixture = RepairFixture::new();
            let claim = fixture.register();
            fixture.send(
                true,
                vec![Replace { values: vec![10] }],
                vec![vec![Replace { values: vec![10] }]],
            );
            fixture.send(true, vec![Push { value: 11 }], vec![vec![Push { value: 11 }]]);
            let claim = fixture.reactivate(claim);
            // Clearing the already empty right side with a nonempty left side consumes a real
            // input but translates to None, unlike merely polling with no queued input.
            fixture.send(false, vec![Clear], vec![]);
            fixture.send(false, vec![], vec![]);
            assert!(fixture.drain_chain().is_empty());
            let mut values = Vec::new();
            fixture.consume_cache(&mut values);
            assert!(
                values.is_empty(),
                "no-op inputs must not fabricate a ready cache snapshot"
            );
            fixture.send(
                true,
                vec![Push { value: 12 }],
                vec![vec![Replace { values: vec![10, 11] }, Push { value: 12 }]],
            );
            fixture.send(true, vec![Push { value: 13 }], vec![vec![Push { value: 13 }]]);
            for _ in 0..3 {
                fixture.consume_cache(&mut values);
            }
            assert_eq!(values, [10, 11, 12, 13]);
            claim.release(fixture.app.world_mut());
        }

        #[test]
        fn branch_only_reactivation_does_not_repair_a_continuously_active_chain() {
            let mut fixture = RepairFixture::new();
            let claim = fixture.register();
            let keeper = fixture
                .chain
                .clone()
                .for_each(|In(_batch): In<Batch>| ())
                .schedule::<ChainConsumer>()
                .register(fixture.app.world_mut());
            fixture.send(
                true,
                vec![Replace { values: vec![10] }],
                vec![vec![Replace { values: vec![10] }]],
            );
            fixture.send(
                false,
                vec![Replace { values: vec![20] }],
                vec![vec![Push { value: 20 }]],
            );
            let mut values = Vec::new();
            fixture.consume_cache(&mut values);
            assert_eq!(values, [10, 20]);
            let identity = claim.entity();
            claim.release(fixture.app.world_mut());
            fixture.send(
                true,
                vec![Push { value: 11 }],
                vec![vec![InsertAt { index: 1, value: 11 }]],
            );
            let claim = fixture.register();
            assert_eq!(claim.entity(), identity);
            assert!(
                fixture.drain_chain().is_empty(),
                "branch activation grants no new chain output"
            );
            fixture.send(false, vec![Push { value: 21 }], vec![vec![Push { value: 21 }]]);
            // The inactive branch missed ordinary output; this is not a demand for a new replay
            // capability or branch-local repair from a chain whose own epoch never changed.
            claim.release(fixture.app.world_mut());
            keeper.release(fixture.app.world_mut());
        }

        #[test]
        fn incremental_only_output_cannot_seed_repair_even_from_an_input_replace() {
            let mut fixture = RepairFixture::new();
            let claim = fixture.register();
            fixture.send(true, vec![Push { value: 10 }], vec![vec![Push { value: 10 }]]);
            fixture.send(false, vec![Push { value: 20 }], vec![vec![Push { value: 20 }]]);
            let claim = fixture.reactivate(claim);
            fixture.send(
                true,
                vec![Push { value: 11 }],
                vec![vec![InsertAt { index: 1, value: 11 }]],
            );
            fixture.send(
                false,
                vec![Replace { values: vec![30] }],
                vec![vec![Pop, Push { value: 30 }]],
            );
            let claim = fixture.reactivate(claim);
            fixture.send(
                true,
                vec![Push { value: 12 }],
                vec![vec![InsertAt { index: 2, value: 12 }]],
            );
            let mut values = Vec::new();
            fixture.consume_cache(&mut values);
            assert!(
                values.is_empty(),
                "incremental-only chain must not fabricate authoritative replay"
            );
            fixture.send(false, vec![Clear], vec![vec![Pop]]);
            fixture.send(
                true,
                vec![Replace { values: vec![40] }],
                vec![vec![Replace { values: vec![40] }]],
            );
            let claim = fixture.reactivate(claim);
            fixture.send(
                true,
                vec![Push { value: 41 }],
                vec![vec![Replace { values: vec![40] }, Push { value: 41 }]],
            );
            for _ in 0..3 {
                fixture.consume_cache(&mut values);
            }
            assert_eq!(
                values,
                [40, 41],
                "an actual output Replace enables subsequent epoch repair"
            );
            claim.release(fixture.app.world_mut());
        }
    }
}
