all notable changes to this project will be documented in this file

the format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://common-changelog.org/), and this project vaguely adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## unreleased

### changed

- upgrade to bevy 0.20
- signals, signal vecs, and signal maps now use move-first delivery by default: ordinary concrete and boxed signal chains do not implement `Clone`, registration (`register` / `try_register`) and combinators consume their receiver, and linear chains support move-only (`!Clone`) items end-to-end
- fan-out is now explicit via `SignalExt::broadcast`, `SignalVecExt::broadcast`, and `SignalMapExt::broadcast` (producing cloneable `Broadcast`, `BroadcastVec`, and `BroadcastMap` handles that perform exact N − 1 clones for N live active dependents)
- replayability is separated from broadcast and expressed via `ReplayableSignal`, `ReplayableSignalVec`, and `ReplayableSignalMap`; non-replayable chains gain current-state replay through construction-time `.cache()` (`CachedSignal`, `CachedSignalVec`, `CachedSignalMap`), with `.cache().broadcast()` sharing one replay cache and `.broadcast().cache()` creating a branch-local warming cache on ordinary event broadcasts
- `jonmo::Builder` no longer implements `Clone`; debug and release builds behave identically, and `child_signal`, `children_signal_vec`, `on_signal*`, and `component_signal` accept move-only values directly (broadcast upstream state before mapping to `Builder` when branching)
- `SignalExt::to_signal_vec` is now purely structural (emitting one `VecDiff::Replace` per upstream `Vec<T>` without implicit deduplication or `Clone` / `PartialEq` bounds); chain `.dedupe().to_signal_vec()` when equality suppression is desired
- `signal::once`, `signal::option`, and `SignalMapExt::map_value` now produce linear move-only signals without implicit replay caching or `Clone` bounds; chain `.cache()` explicitly before replay-only consumers such as `switch` or `switch_signal_map`
- dynamic switching combinators (`flatten`, `switch`, `switch_signal_vec`, `switch_signal_map`, `map_signal`, `filter_signal`, `map_value_signal`) consume inner replayable signals once without requiring inner signal `Clone` bounds, and track warming candidates and multi-slot replacement cohorts without polling
- signal lifecycle ownership uses world-local cleanup and linear `SignalClaim` / entity-owned `SignalClaims` tokens instead of shared cloneable handles

### added

- `SignalVecExt::flatten` for flattening a `SignalVec` of replayable inner `SignalVec`s with incremental offset-translated diffs and atomic warming replacement cohorts
- `SignalClaim`, `Scheduled`, `ScheduledVec`, `ScheduledMap`, `VecDiff`, and `MapDiff` in `jonmo::prelude` (while scoping custom combinator and runtime authoring APIs under `jonmo::registration`, `jonmo::dynamic`, `jonmo::graph`, and crate root)
- public combinator-authoring and runtime capabilities: `SignalRegistrar`, `SignalMaterializer`, branded `SignalEntityRef`, `OutputDelivery` (`move_only` and `broadcast`), `ReplayInitialization`, and `SignalRegistrationError`
- declarative managed dynamic operator specification `DynamicSignalOperatorSpec` (`scalar`, `indexed`, `keyed`) with ordinary move-first `scalar_output` / `signal_vec_output` / `signal_map_output` reducers (`!Clone` queues and outputs) and managed `Warming` `replayable_scalar_output` / `replayable_signal_vec_output` / `replayable_signal_map_output` reducers
- non-executing replay-cache inspection helpers `inspect_scalar_replay_cache`, `inspect_vec_replay_cache`, `inspect_map_replay_cache`, and `ReplayCacheInspection`
- opt-in `imperative-signal-evaluation` feature exposing `jonmo::graph::{run_signal_subgraph_once, BoxedSignalValue, downcast_signal_value}` outside the normal prelude

### fixed

- cross-world `Entity` cleanup corruption by replacing process-global cleanup queues with world-local `WorldCleanupQueue` routing and entity despawn cleanup
- `MutableVec` and `MutableBTreeMap` initial-empty, pre-registration mutation, post-registration pre-first-replay mutation, and multi-subscriber replay snapshot delivery bugs
- `SignalVecExt::sort_by` and `SignalVecExt::sort_by_key` backward equal-key `Move` and post-move `UpdateAt` index bugs
- `SignalVecExt::enumerate` and `SignalVecExt::intersperse` unbounded index table growth and foreign-world `Entity` lookup collisions
- skipped system parameter validation (`RunSystemError::Skipped`) falsely emitting `None` or silently dropping collection updates across `SignalExt::filter`, `map_bool`, `map_true`, `map_false`, `map_option`, `map_some`, `map_none`, `SignalVecExt::map`, and `SignalMapExt::map_value`
- dynamic operator `.schedule::<S>()` propagation to runtime adapters and untagged inner chains, mid-pass topological level promotion when switching to deeper inner chains, and `1 -> 0 -> 1` unretained reactivation state resets
- `Builder::spawn_on_entity` child block index collisions when composing multiple builders onto one entity, and panics when reactive children are externally despawned
- `JonmoPlugin::registration_recursion_limit(0)` falsely panicking on single-pass schedules when no recursive registrations occur

### removed

- cloneable `SignalHandle` / `SignalHandles` and raw `SignalSystem(Entity)` wrapper surfaces (replaced by linear `SignalClaim`, entity-owned `SignalClaims`, and generational `SignalEntity`)
- default-feature `poll_signal` / `AnyClone` / `SignalDynClone` surfaces and the `dyn-clone` dependency
- invalid `Clone` implementation on `jonmo::Builder`

# 0.7.0 (2026-02-09)

### changed

- upgrade to bevy 0.18

# 0.6.0 (2026-02-09)

### changed

- signal graph processed in deterministic topological order instead of the previous nondeterministic depth first order
- `SignalBuilder::*` methods moved to free functions in the `signal` module (e.g. `SignalBuilder::always` -> `signal::always`)
- `JonmoBuilder` renamed to `jonmo::Builder`
- `jonmo::Builder` is now lock-free
- `jonmo::Builder::hold_signals` renamed to `jonmo::Builder::hold_tasks` and takes `Box<dyn SignalTask>`s instead of `SignalHandles`
- renamed `SignalExt::combine` to `SignalExt::zip`
- `MutableVecBuilder`/`MutableBTreeMapBuilder` changed to `MutableVec::builder()`/`MutableBTreeMap::builder()` with simple chainable `.values` and `.with_values` methods
- removed self item `Clone` bound from `SignalVecExt::map_signal` and `SignalVecExt::filter_map`
- `.get_mut`s unwrapped and `.get_entity_mut`s changed to `.entity_mut` in cases where they were silently ignoring invariant violations

### added
- `SignalExt/SignalVecExt/SignalMapExt::schedule`, enabling granular control of which schedule each node in the signal graph runs during
- `SignalExt::take`
- `SignalExt::skip`
- `signal::once`
- `signal::from_component_changed`
- `signal::from_resource_changed`
- `signal::zip!`, a variadic flattened version of `SignalExt::zip`
- `track_caller` derive for panicking `LazyEntity` methods
- panic (debug only) or error log that cloning `jonmo::Builder`s at runtime is a bug

### fixed

- deadlock when despawning `MutableVec/BTreeMap`s during another `MutableVec/BTreeMap` despawn
- initially empty `MutableVec/BTreeMap`s work as expected when output to `.switch_signal_vec/map`
- `SignalVecExt::debug` and `SignalMapExt::debug` now log correct code location

### removed
- `*_lazy` signal builder functions, the non-`lazy` versions now take both `Entity` and `LazyEntity`
- `jonmo::Builder::signal_from_*` methods, use corresponding `signal` building functions with `.task()` and `jonmo::Builder::hold_tasks` instead
- `jonmo::Builder::component_signal_from_*` methods, use corresponding `signal` building functions with `jonmo::Builder::component_signal` instead

# 0.5.0 (2025-12-19)

### changed

- `.entity_sync` renamed to `.lazy_entity`
- `SignalExt::combine` emits its latest upstream outputs every frame, unlike previously, when it only emitted on frames where the latest output pair was yet to be emitted

### added

- `JonmoBuilder::hold_signals`
- `JonmoBuilder::on_despawn`
- `SignalBuilder::from_function`
- `SignalBuilder::always`
- `signal::option`
- `SignalExt::map_bool_in`
- `SignalExt::map_true_in`
- `SignalExt::map_false_in`
- `SignalExt::map_option_in`
- `SignalExt::map_some_in`
- `SignalExt::map_none_in`
- `signal::eq!`
- `signal::all!`
- `signal::any!`
- `signal::distinct!`
- `signal::sum!`
- `signal::product!`
- `signal::min!`
- `signal::max!`

### fixed

- `SignalExt::debug` now logs correct code location

# 0.4.2 (2025-11-28)

### added

- allow specifying custom schedule for the `JonmoPlugin`

# 0.3.2 (2025-11-28)

### added

- allow specifying custom schedule for the `JonmoPlugin`

# 0.4.1 (2025-11-27)

### fixed

- initially empty `MutableVec`s and `MutableBTreeMap`s won't double emit their first additions

# 0.3.1 (2025-11-27)

### fixed

- initially empty `MutableVec`s and `MutableBTreeMap`s won't double emit their first additions

# 0.4.0 (2025-11-20)

### changed

- upgraded to Bevy 0.17

### added

- `JonmoBuilder::component_signal_from_component_changed` convenience method
- `SignalExt::component_changed`
- `SignalExt::switch_signal_map`

### fixed

- `.switch_signal_vec` can take signal vecs that were initialized outside the body of the passed function
- `.flatten`-ing signal combinators don't leak helper entities

# 0.3.0 (2025-11-19)

### changed

- reactive collections `MutableVec` and `MutableBTreeMap` are now lock-free and managed by entities and components
- `Signal*Clone` renamed to `Signal*DynClone`

### added

- `SignalProcessing` `SystemSet` for scheduling systems around signal processing
- `MutableVec` and `MutableBTreeMap` can be built from with `FromWorld`, `&mut World`, and `&mut Commands`
- `MutableVecBuilder` and `MutableBTreeMapBuilder`
- `JonmoBuilder::on_spawn_with_system` convenience method
- `JonmoBuilder::observe` convenience method
- `JonmoBuilder::on_signal_with_entity` convenience method
- deref convenience for `LazyEntity`
- `JonmoBuilder` derives `Default`

