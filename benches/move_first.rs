//! Release benchmark suite for Move-First Signal Delivery
//! (`docs/move-first-signal-delivery-plan.md`).
//!
//! Measures normal schedule-driven signal execution (never invoking `imperative-signal-evaluation`)
//! and verifies exact clone-count invariants across:
//! 1. 16-stage linear scalar chain carrying a small clone-counted value (`0` clones).
//! 2. 16-stage linear scalar chain carrying a 4 KiB `!Clone` payload (`0` clones).
//! 3. Broadcast fan-out with `2`, `8`, and `64` live active dependents (`N - 1` clones/update).
//! 4. Collection diff propagation through `MutableVec` + linear `SignalVecExt::map_in`.
//! 5. `Builder`-valued `SignalVec` updates (`children_signal_vec`).
//! 6. Replay-cache warming latency and steady-state revision propagation across schedules.
//! 7. Dynamic pending scalar replacement and multi-child `IndexedSlots` cohort
//!    completion/cancellation.

#![allow(missing_docs, clippy::std_instead_of_core, clippy::std_instead_of_alloc)]

extern crate alloc;

use alloc::{boxed::Box, collections::VecDeque, format, string::String, sync::Arc, vec, vec::Vec};
use bevy_app::{App, Last};
use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use bevy_platform::sync::Mutex;
use core::{
    hint::black_box,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};
use jonmo::{
    DynamicEmissions, DynamicManagerUpdate, DynamicSignalOperatorSpec, IndexedSlotCommand, OutputDelivery,
    ReplayCacheInspection, SameSignalIdentity, ScalarSignalMaterializer, ScalarSlotCommand,
    inspect_scalar_replay_cache,
    prelude::*,
    signal::{self, BoxedReplayableSignal},
    signal_vec::{MutableVec, SignalVecExt, VecDiff},
};
use std::time::{Duration, Instant};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash, Default)]
struct ProducerSchedule;

#[derive(Debug)]
struct CountedSmall {
    value: u64,
    clones: Arc<AtomicUsize>,
}

impl CountedSmall {
    fn new(value: u64, clones: &Arc<AtomicUsize>) -> Self {
        Self {
            value,
            clones: Arc::clone(clones),
        }
    }
}

impl Clone for CountedSmall {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::Relaxed);
        Self {
            value: self.value,
            clones: Arc::clone(&self.clones),
        }
    }
}

/// Strictly `!Clone` 4 KiB payload for linear zero-copy transport benchmarking.
struct LargeMoveOnly {
    bytes: Box<[u8; 4096]>,
    label: String,
    checksum: u64,
}

impl LargeMoveOnly {
    fn new(seed: u64) -> Self {
        let mut bytes = Box::new([0_u8; 4096]);
        bytes[0] = (seed & 0xff) as u8;
        bytes[2048] = ((seed >> 8) & 0xff) as u8;
        bytes[4095] = ((seed >> 16) & 0xff) as u8;
        Self {
            bytes,
            label: String::from("move-first-payload"),
            checksum: seed,
        }
    }
}

struct CustomScalar<T: 'static>(ScalarSignalMaterializer<T>);

impl<T: 'static> Signal for CustomScalar<T> {
    type Item = T;

    fn into_signal_materializer(self: Box<Self>) -> ScalarSignalMaterializer<Self::Item> {
        self.0
    }
}

fn configured_iters(default: usize) -> usize {
    std::env::var("JONMO_BENCH_ITERS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&iters| iters > 0)
        .unwrap_or(default)
}

fn report(name: &str, iters: usize, elapsed: Duration, note: &str) {
    let total_ns = elapsed.as_nanos();
    let ns_per_iter = total_ns as f64 / iters as f64;
    println!(
        "{name:<48} {iters:>6} iters | {ns_per_iter:>10.1} ns/iter | total {:>7.2} ms | {note}",
        elapsed.as_secs_f64() * 1000.0
    );
}

fn bench_linear_small_chain_16(iters: usize) {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    let clones = Arc::new(AtomicUsize::new(0));
    let sink_sum = Arc::new(AtomicU64::new(0));
    let counter = Arc::new(AtomicU64::new(0));

    let mut chain: BoxedSignal<CountedSmall> = signal::from_system({
        let clones = Arc::clone(&clones);
        let counter = Arc::clone(&counter);
        move |In(())| {
            let next = counter.fetch_add(1, Ordering::Relaxed) + 1;
            Some(CountedSmall::new(next, &clones))
        }
    })
    .boxed();

    for step in 0..16_u64 {
        chain = chain
            .map_in(move |mut item: CountedSmall| {
                item.value = item.value.wrapping_add(step + 1);
                item
            })
            .boxed();
    }

    let claim = chain
        .map({
            let sink_sum = Arc::clone(&sink_sum);
            move |In(item): In<CountedSmall>| {
                sink_sum.fetch_add(black_box(item.value), Ordering::Relaxed);
            }
        })
        .register(app.world_mut());

    for _ in 0..32 {
        app.update();
    }
    clones.store(0, Ordering::SeqCst);

    let start = Instant::now();
    for _ in 0..iters {
        app.update();
    }
    let elapsed = start.elapsed();

    let observed_clones = clones.load(Ordering::SeqCst);
    assert_eq!(observed_clones, 0, "linear 16-stage small chain must perform 0 clones");
    black_box(sink_sum.load(Ordering::Relaxed));
    claim.release(app.world_mut());

    report(
        "linear_chain_16_small_u64",
        iters,
        elapsed,
        "clones/iter = 0 (verified)",
    );
}

fn bench_linear_large_non_clone_chain_16(iters: usize) {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    let sink_sum = Arc::new(AtomicU64::new(0));
    let counter = Arc::new(AtomicU64::new(0));

    let mut chain: BoxedSignal<LargeMoveOnly> = signal::from_system({
        let counter = Arc::clone(&counter);
        move |In(())| {
            let next = counter.fetch_add(1, Ordering::Relaxed) + 1;
            Some(LargeMoveOnly::new(next))
        }
    })
    .boxed();

    for step in 0..16_usize {
        chain = chain
            .map_in(move |mut item: LargeMoveOnly| {
                let idx = (step * 251) % 4096;
                item.bytes[idx] = item.bytes[idx].wrapping_add(1);
                item.checksum = item.checksum.wrapping_add(u64::from(item.bytes[idx]));
                item
            })
            .boxed();
    }

    let claim = chain
        .map({
            let sink_sum = Arc::clone(&sink_sum);
            move |In(item): In<LargeMoveOnly>| {
                let total = item.checksum.wrapping_add(item.label.len() as u64);
                sink_sum.fetch_add(black_box(total), Ordering::Relaxed);
            }
        })
        .register(app.world_mut());

    for _ in 0..32 {
        app.update();
    }

    let start = Instant::now();
    for _ in 0..iters {
        app.update();
    }
    let elapsed = start.elapsed();

    black_box(sink_sum.load(Ordering::Relaxed));
    claim.release(app.world_mut());

    report(
        "linear_chain_16_large_4kib_non_clone",
        iters,
        elapsed,
        "!Clone 4 KiB moved in-place",
    );
}

fn bench_broadcast_fanout(dependents: usize, iters: usize) {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    let clones = Arc::new(AtomicUsize::new(0));
    let counter = Arc::new(AtomicU64::new(0));
    let sink_sum = Arc::new(AtomicU64::new(0));

    let shared = signal::from_system({
        let clones = Arc::clone(&clones);
        let counter = Arc::clone(&counter);
        move |In(())| {
            let next = counter.fetch_add(1, Ordering::Relaxed) + 1;
            Some(CountedSmall::new(next, &clones))
        }
    })
    .broadcast();

    let mut claims = Vec::with_capacity(dependents);
    for slot in 0..dependents {
        let sink_sum = Arc::clone(&sink_sum);
        claims.push(
            shared
                .clone()
                .map(move |In(item): In<CountedSmall>| {
                    sink_sum.fetch_add(black_box(item.value.wrapping_add(slot as u64)), Ordering::Relaxed);
                })
                .register(app.world_mut()),
        );
    }

    for _ in 0..16 {
        app.update();
    }
    clones.store(0, Ordering::SeqCst);

    let start = Instant::now();
    for _ in 0..iters {
        app.update();
    }
    let elapsed = start.elapsed();

    let observed_clones = clones.load(Ordering::SeqCst);
    let expected_clones = (dependents - 1) * iters;
    assert_eq!(
        observed_clones, expected_clones,
        "broadcast with {dependents} dependents must perform exact (N - 1) clones per update"
    );
    black_box(sink_sum.load(Ordering::Relaxed));

    for claim in claims {
        claim.release(app.world_mut());
    }
    drop(shared);

    report(
        &format!("broadcast_fanout_{dependents}_dependents"),
        iters,
        elapsed,
        &format!("clones/iter = {} (exact N - 1)", dependents - 1),
    );
}

fn bench_collection_diff_propagation(iters: usize) {
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    let vec_source = MutableVec::builder()
        .values((0..32_u64).collect::<Vec<_>>())
        .spawn(app.world_mut());
    let diff_count = Arc::new(AtomicUsize::new(0));

    let claim = vec_source
        .signal_vec()
        .map_in(LargeMoveOnly::new)
        .for_each({
            let diff_count = Arc::clone(&diff_count);
            move |In(diffs): In<Vec<VecDiff<LargeMoveOnly>>>| {
                for diff in diffs {
                    match diff {
                        VecDiff::Replace { values } => {
                            diff_count.fetch_add(black_box(values.len()), Ordering::Relaxed);
                        }
                        VecDiff::InsertAt { value, .. } | VecDiff::UpdateAt { value, .. } | VecDiff::Push { value } => {
                            diff_count.fetch_add(black_box(value.bytes[0] as usize + 1), Ordering::Relaxed);
                        }
                        VecDiff::RemoveAt { .. } | VecDiff::Move { .. } | VecDiff::Pop | VecDiff::Clear => {
                            diff_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        })
        .register(app.world_mut());

    app.update();

    let start = Instant::now();
    for step in 0..iters {
        {
            let mut guard = vec_source.write(app.world_mut());
            guard.push(step as u64 + 100);
            guard.set((step * 7) % 32, step as u64 + 200);
            guard.pop();
        }
        app.update();
    }
    let elapsed = start.elapsed();

    black_box(diff_count.load(Ordering::Relaxed));
    claim.release(app.world_mut());

    report(
        "collection_diff_mutable_vec_to_large_non_clone",
        iters,
        elapsed,
        "push + update + pop / iter",
    );

    let clone_retaining_claim = vec_source
        .signal_vec()
        .filter(|In(value): In<u64>| value % 2 == 0)
        .cache()
        .for_each({
            let diff_count = Arc::clone(&diff_count);
            move |In(diffs): In<Vec<VecDiff<u64>>>| {
                diff_count.fetch_add(black_box(diffs.len()), Ordering::Relaxed);
            }
        })
        .register(app.world_mut());

    app.update();

    let clone_start = Instant::now();
    for step in 0..iters {
        {
            let mut guard = vec_source.write(app.world_mut());
            guard.push(step as u64 + 300);
            guard.set((step * 7) % 32, step as u64 + 400);
            guard.pop();
        }
        app.update();
    }
    let clone_elapsed = clone_start.elapsed();

    black_box(diff_count.load(Ordering::Relaxed));
    clone_retaining_claim.release(app.world_mut());

    report(
        "collection_diff_filter_plus_cache_clone_heavy",
        iters,
        clone_elapsed,
        "filter + cache push/update/pop / iter",
    );
}

#[cfg(feature = "builder")]
fn bench_builder_signal_vec_updates(iters: usize) {
    #[derive(Component)]
    struct BenchRow(u64);

    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    let rows = MutableVec::builder()
        .values((0..16_u64).collect::<Vec<_>>())
        .spawn(app.world_mut());
    let parent = jonmo::Builder::new()
        .children_signal_vec(
            rows.signal_vec()
                .map_in(|value: u64| jonmo::Builder::new().insert(BenchRow(black_box(value)))),
        )
        .spawn(app.world_mut());

    app.update();

    let start = Instant::now();
    for step in 0..iters {
        {
            let mut guard = rows.write(app.world_mut());
            guard.push(step as u64 + 1000);
            guard.set(step % 16, step as u64 + 2000);
            guard.remove(0);
        }
        app.update();
    }
    let elapsed = start.elapsed();

    let children = app.world().get::<Children>(parent).expect("parent has children");
    assert_eq!(children.len(), 16);
    let total_row_sum: u64 = children
        .iter()
        .map(|child| app.world().get::<BenchRow>(child).expect("child has BenchRow").0)
        .sum();
    black_box(total_row_sum);
    app.world_mut().entity_mut(parent).despawn();
    app.update();

    report(
        "builder_children_signal_vec_updates",
        iters,
        elapsed,
        "push + update + remove(0) / iter",
    );
}

fn bench_replay_cache_warming_and_cross_schedule_revision(iters: usize) {
    // Part A: fresh `.cache()` warming + first authoritative snapshot latency across schedules.
    let warm_iters = (iters / 4).max(50);
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default().with_schedule::<ProducerSchedule>());

    let warm_start = Instant::now();
    for step in 0..warm_iters {
        let cached = signal::once(step as u64).schedule::<ProducerSchedule>().cache();
        let claim = cached.register(app.world_mut());
        let cache_entity = claim.entity();
        assert_eq!(
            inspect_scalar_replay_cache::<u64>(app.world(), cache_entity),
            Some(ReplayCacheInspection::Warming)
        );
        app.world_mut().run_schedule(ProducerSchedule);
        app.world_mut().run_schedule(Last);
        assert!(matches!(
            inspect_scalar_replay_cache::<u64>(app.world(), cache_entity),
            Some(ReplayCacheInspection::Ready { current, revision: 1 }) if current == step as u64
        ));
        claim.release(app.world_mut());
    }
    let warm_elapsed = warm_start.elapsed();
    report(
        "replay_cache_warming_cross_schedule",
        warm_iters,
        warm_elapsed,
        "Warming -> ProducerSchedule -> Ready",
    );

    // Part B: steady-state cross-schedule cached revision propagation.
    let counter = Arc::new(AtomicU64::new(0));
    let sink_sum = Arc::new(AtomicU64::new(0));
    let steady_claim = signal::from_system({
        let counter = Arc::clone(&counter);
        move |In(())| Some(counter.fetch_add(1, Ordering::Relaxed) + 1)
    })
    .schedule::<ProducerSchedule>()
    .cache()
    .map({
        let sink_sum = Arc::clone(&sink_sum);
        move |In(v): In<u64>| {
            sink_sum.fetch_add(black_box(v), Ordering::Relaxed);
        }
    })
    .register(app.world_mut());

    app.world_mut().run_schedule(ProducerSchedule);
    app.world_mut().run_schedule(Last);

    let steady_start = Instant::now();
    for _ in 0..iters {
        app.world_mut().run_schedule(ProducerSchedule);
        app.world_mut().run_schedule(Last);
    }
    let steady_elapsed = steady_start.elapsed();
    black_box(sink_sum.load(Ordering::Relaxed));
    steady_claim.release(app.world_mut());

    report(
        "replay_cache_steady_state_cross_schedule",
        iters,
        steady_elapsed,
        "ProducerSchedule -> Last cached hop",
    );
}

fn bench_dynamic_pending_scalar_and_cohort_transitions(iters: usize) {
    type ScalarUpdate = DynamicManagerUpdate<ScalarSlotCommand<BoxedReplayableSignal<u64>>, u64>;
    type IndexedUpdate = DynamicManagerUpdate<IndexedSlotCommand<BoxedReplayableSignal<u64>>, (usize, u64)>;

    let dynamic_iters = (iters / 2).max(100);
    let mut app = App::new();
    app.add_plugins(JonmoPlugin::default());

    let scalar_updates: Arc<Mutex<VecDeque<ScalarUpdate>>> = Arc::new(Mutex::new(VecDeque::new()));
    let indexed_updates: Arc<Mutex<VecDeque<IndexedUpdate>>> = Arc::new(Mutex::new(VecDeque::new()));
    let sink_sum = Arc::new(AtomicU64::new(0));

    let scalar_claim = CustomScalar(ScalarSignalMaterializer::new({
        let scalar_updates = Arc::clone(&scalar_updates);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| scalar_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::scalar(manager)
                    .replay_scalar::<u64>()
                    .adapter(|(), v| DynamicEmissions::One(v))
                    .scalar_output(|mut q| q.pop())
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        }
    }))
    .map({
        let sink_sum = Arc::clone(&sink_sum);
        move |In(v): In<u64>| {
            sink_sum.fetch_add(black_box(v), Ordering::Relaxed);
        }
    })
    .register(app.world_mut());

    let indexed_claim = CustomScalar(ScalarSignalMaterializer::new({
        let indexed_updates = Arc::clone(&indexed_updates);
        move |registrar| {
            let manager = registrar.signal_entity(
                (),
                move |In(())| indexed_updates.lock().unwrap().pop_front(),
                OutputDelivery::move_only(),
            )?;
            registrar.dynamic_operator(
                DynamicSignalOperatorSpec::indexed(manager)
                    .replay_scalar::<u64>()
                    .adapter(|idx, v| DynamicEmissions::One((idx, v)))
                    .scalar_output(|q| (!q.is_empty()).then_some(q))
                    .same_signal_identity(SameSignalIdentity::Restart),
            )
        }
    }))
    .map({
        let sink_sum = Arc::clone(&sink_sum);
        move |In(batch): In<Vec<(usize, u64)>>| {
            for (idx, v) in batch {
                sink_sum.fetch_add(black_box(v.wrapping_add(idx as u64)), Ordering::Relaxed);
            }
        }
    })
    .register(app.world_mut());

    let start = Instant::now();
    for step in 0..dynamic_iters {
        let base = step as u64;
        if step % 2 == 0 {
            // Stage a pending warming scalar replacement AND a 2-child cohort that will both be
            // superseded and cancelled on the odd step!
            scalar_updates.lock().unwrap().push_back(ScalarUpdate::new(
                [ScalarSlotCommand::Select {
                    signal: signal::from_system(|In(())| None::<u64>).cache().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ));
            indexed_updates.lock().unwrap().push_back(IndexedUpdate::new(
                [IndexedSlotCommand::ReplaceAll {
                    signals: vec![
                        signal::once(base + 10).cache().boxed_replayable(),
                        signal::from_system(|In(())| None::<u64>).cache().boxed_replayable(),
                    ],
                }],
                DynamicEmissions::None,
            ));
        } else {
            // Supersede and cancel the pending scalar + warming cohort with ready replacements!
            scalar_updates.lock().unwrap().push_back(ScalarUpdate::new(
                [ScalarSlotCommand::Select {
                    signal: signal::once(base + 1).cache().boxed_replayable(),
                }],
                DynamicEmissions::None,
            ));
            indexed_updates.lock().unwrap().push_back(IndexedUpdate::new(
                [IndexedSlotCommand::ReplaceAll {
                    signals: vec![
                        signal::once(base + 1).cache().boxed_replayable(),
                        signal::once(base + 2).cache().boxed_replayable(),
                        signal::once(base + 3).cache().boxed_replayable(),
                        signal::once(base + 4).cache().boxed_replayable(),
                    ],
                }],
                DynamicEmissions::None,
            ));
        }
        app.update();
    }
    let elapsed = start.elapsed();

    black_box(sink_sum.load(Ordering::Relaxed));
    scalar_claim.release(app.world_mut());
    indexed_claim.release(app.world_mut());
    app.update();

    report(
        "dynamic_scalar_replace_and_cohort_cancel_commit",
        dynamic_iters,
        elapsed,
        "scalar replace + 2-slot cancel / 4-slot commit",
    );
}

fn main() {
    let iters = configured_iters(1_000);
    println!("=== jonmo move-first release benchmark suite (base iters = {iters}) ===");
    bench_linear_small_chain_16(iters);
    bench_linear_large_non_clone_chain_16(iters);
    bench_broadcast_fanout(2, iters);
    bench_broadcast_fanout(8, iters);
    bench_broadcast_fanout(64, iters);
    bench_collection_diff_propagation(iters);
    #[cfg(feature = "builder")]
    bench_builder_signal_vec_updates(iters);
    bench_replay_cache_warming_and_cross_schedule_revision(iters);
    bench_dynamic_pending_scalar_and_cohort_transitions(iters);
}
