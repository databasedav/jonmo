all notable changes to this project will be documented in this file

the format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://common-changelog.org/), and this project vaguely adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## unreleased

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

