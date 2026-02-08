all notable changes to this project will be documented in this file

the format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://common-changelog.org/), and this project vaguely adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## unreleased

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

