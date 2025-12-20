all notable changes to this project will be documented in this file

the format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://common-changelog.org/), and this project vaguely adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## unreleased

# 0.6.0-rc.1 (2025-12-19)

### changed

- upgraded to bevy 0.18-rc.1

# 0.5.0 (2025-12-19)

### changed

- `.entity_sync` renamed to `.lazy_entity`
- `SignalExt::combine` always `.clone`s its latest upstream outputs instead of `.take`-ing them

### added

- `JonmoBuilder::hold_signals`
- `JonmoBuilder::on_despawn`
- `SignalBuilder::from_function`
- `SignalBuilder::always`
- `SignalExt::map_bool_in`
- `SignalExt::map_true_in`
- `SignalExt::map_false_in`
- `SignalExt::map_option_in`
- `SignalExt::map_some_in`
- `SignalExt::map_none_in`
- `signal::option`
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

- upgraded to bevy 0.17

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

