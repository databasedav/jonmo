all notable changes to this project will be documented in this file

the format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://common-changelog.org/), and this project vaguely adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## unreleased

### changed

- upgraded to Bevy 0.17

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

