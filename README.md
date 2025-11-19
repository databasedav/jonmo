# jonmo [জন্ম](https://translate.google.com/?sl=bn&tl=en&text=%E0%A6%9C%E0%A6%A8%E0%A7%8D%E0%A6%AE&op=translate)

[![Crates.io Version](https://img.shields.io/crates/v/jonmo?style=for-the-badge)](https://crates.io/crates/jonmo)
[![Docs.rs](https://img.shields.io/docsrs/jonmo?style=for-the-badge)](https://docs.rs/jonmo)
[![Following released Bevy versions](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue?style=for-the-badge)](https://bevyengine.org/learn/quick-start/plugin-development/#main-branch-tracking)

```text
in bengali, jonmo means "birth"
```

[jonmo](https://github.com/databasedav/jonmo) provides an ergonomic, functional, and declarative API for specifying Bevy [system](https://docs.rs/bevy/latest/bevy/ecs/system/index.html) dependency graphs, where "output" handles to nodes of the graph are canonically referred to as "signals". Building upon these signals, jonmo offers a high level [entity builder](https://docs.rs/jonmo/latest/jonmo/builder/struct.JonmoBuilder.html) which enables one to declare reactive entities, components, and children using a familiar fluent syntax with semantics and API ported from the incredible [FRP](https://en.wikipedia.org/wiki/Functional_reactive_programming) signals of [futures-signals](https://github.com/Pauan/rust-signals) and its web UI dependents [MoonZoon](https://github.com/MoonZoon/MoonZoon) and [Dominator](https://github.com/Pauan/rust-dominator).

The runtime of jonmo is quite simple; every frame, the outputs of systems are forwarded to their dependants, recursively. The complexity and power of jonmo really emerges from its monadic signal combinators, defined within the [`SignalExt`](https://docs.rs/jonmo/latest/jonmo/signal/trait.SignalExt.html), [`SignalVecExt`](https://docs.rs/jonmo/latest/jonmo/signal_vec/trait.SignalVecExt.html), and [`SignalMapExt`](https://docs.rs/jonmo/latest/jonmo/signal_map/trait.SignalMapExt.html) traits (ported from futures-signals' traits of the same name), which internally manage special Bevy systems that allow for the declarative composition of complex data flows with minimalistic, high-level, signals-oriented methods.

### assorted features:
- fine-grained reactivity for all entities, components, and children
- ***diff-less*** constant-time reactive updates for collections (available through [`MutableVec`](https://docs.rs/jonmo/latest/jonmo/signal_vec/struct.MutableVec.html) and [`MutableBTreeMap`](https://docs.rs/jonmo/latest/jonmo/signal_map/struct.MutableBTreeMap.html))
- automated system lifecycle management when using the builder API, simple component on-remove hook when not
- polling API for when one needs an escape hatch from the regular push-based output semantics (polling is used sparsely internally for some combinators)
- either wrappers (a la <https://github.com/rayon-rs/either>) and type-erased signals (via boxing) for cheap and flexible management of distinct signal types from different branches of logic
- `no_std` compatible

## [feature flags](https://docs.rs/jonmo/latest/jonmo/#feature-flags-1)

## examples
<p align="center">
  <img src="https://raw.githubusercontent.com/databasedav/jonmo/main/docs/static/counter.gif">
</p>

```rust,ignore
//! Simple counter example, ported from a similar example in haalka.
//!
//! This example introduces some concepts for building reactive Bevy applications using jonmo's
//! signals:
//!
//! 1. **World-Driven State**: The application's state (the counter's value) is stored in a standard
//!    Bevy `Component`, the "single source of truth".
//!
//! 2. **Declarative UI with `JonmoBuilder`**: The entire entity hierarchy is defined up-front in a
//!    clean, colocated, declarative style.
//!
//! 3. **The `LazyEntity` Pattern**: related entities can refer to the state-holding entity *before*
//!    it has been spawned, solving a common ordering problem in hierarchical entity construction.
//!
//! 4. **Reactivity with `.signal` methods**: A signal is created that reads the state component,
//!    whose output is then used to reactively update other components, like the `Text` for the
//!    counter's display.

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, JonmoPlugin))
        .add_systems(
            Startup,
            (
                |world: &mut World| {
                    ui_root().spawn(world);
                },
                camera,
            ),
        )
        .run();
}

#[derive(Component, Clone, Deref, DerefMut)]
struct Counter(i32);

fn ui_root() -> JonmoBuilder {
    // --- The `LazyEntity` Pattern ---
    // We need the buttons and the text display to know the `Entity` ID of the node
    // that will hold the `Counter` component. However, that node hasn't been spawned yet.
    // `LazyEntity` acts as a placeholder or a "promise" for an entity that will be
    // spawned by a `JonmoBuilder` later. It can be cloned and passed around freely, e.g. into the
    // bodies of signal systems.
    let counter_holder = LazyEntity::new();

    JonmoBuilder::from(Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    })
    .child(
        JonmoBuilder::from(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(15.0),
            align_items: AlignItems::Center,
            padding: UiRect::all(Val::Px(25.)),
            ..default()
        })
        .insert(Counter(0))
        // --- Fulfilling the Promise ---
        // `.entity_sync()` connects the `LazyEntity` to the actual entity that this `JonmoBuilder` spawns. Now calling
        // `counter_holder.get()` from other deferred contexts, e.g. in the bodies of signal systems, will return the
        // `Entity` ID of this row node.
        .entity_sync(counter_holder.clone())
        .child(counter_button(counter_holder.clone(), PINK, "-", -1))
        .child(
            JonmoBuilder::from((Node::default(), TextFont::from_font_size(25.)))
                // --- Reactivity ---
                // `component_signal` is a core fixture of jonmo's reactivity. It takes a signal as an argument and
                // uses its output to insert or update a component on the entity being built. Here,
                // we're creating a signal that will produce a `Text` component whenever the counter
                // changes.
                .component_signal(
                    // `SignalBuilder::from_component_lazy` creates a signal that reactively reads a component from an
                    // entity that doesn't exist yet, identified by our `LazyEntity`.
                    SignalBuilder::from_component_lazy(counter_holder.clone())
                        // Rust's type system is intelligent enough to infer the "from_component" target from the
                        // definition of the system passed to any of jonmo's signal combinators. `map_in` is a shorthand
                        // for `.map` which unpacks its `In` input; this is especially convenient when the system passed
                        // to `.map` does not need any other `SystemParam`s. Here, we simply deref the inner counter
                        // value from the fetched `Counter` component for further transformation.
                        .map_in(|counter: Counter| *counter)
                        // `.dedupe()` ensures the rest of the chain only runs when the counter's value *actually
                        // changes*, preventing redundant updates every frame.
                        .dedupe()
                        // `Text` expects a `String` and `.to_string` expects a reference
                        .map_in_ref(ToString::to_string)
                        .map_in(Text)
                        // `component_signal` expects an `Option<Component>`. If the signal produces `None`, the
                        // component is removed. Here, our value is always present, so we wrap it in `Some`.
                        .map_in(Some),
                ),
        )
        .child(counter_button(counter_holder, BLUE, "+", 1)),
    )
}

fn counter_button(counter_holder: LazyEntity, color: Color, label: &'static str, step: i32) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            width: Val::Px(45.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BorderRadius::MAX,
        BackgroundColor(color),
    ))
    // `.with_entity()` provides a convenient hook to run code on this builder's `EntityWorldMut`, e.g. for setting up
    // observers
    .with_entity(move |mut entity| {
        entity.observe(
            move |trigger: Trigger<Pointer<Click>>, mut counters: Query<&mut Counter>| {
                if matches!(trigger.button, PointerButton::Primary) {
                    // Use the fulfilled `LazyEntity` to get mutable access to the `Counter` component on our
                    // state-holding entity.
                    if let Ok(mut counter) = counters.get_mut(counter_holder.get()) {
                        // --- State Mutation ---
                        // Because our text display has a signal that reads this component, this change will
                        // automatically trigger a UI update at the end of the frame.
                        **counter += step;
                    }
                }
            },
        );
    })
    .child(JonmoBuilder::from((Text::from(label), TextFont::from_font_size(25.))))
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
```

### on the web

All examples are compiled to wasm for both webgl2 and webgpu (check [compatibility](<https://github.com/gpuweb/gpuweb/wiki/Implementation-Status#implementation-status>)) and deployed to github pages.

- [**`basic`**](https://github.com/databasedav/jonmo/blob/main/examples/basic.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/basic/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/basic/)

    a simple increasing timer without using the entity builder, showcasing the least invasive way to start using jonmo signals in existing Bevy apps

- [**`basic_builder_inject`**](https://github.com/databasedav/jonmo/blob/main/examples/basic_builder_inject.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/basic_builder_inject/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/basic_builder_inject/)

    a simple increasing timer injecting the entity builder into an existing entity, showcasing a less invasive way to start using jonmo signals in existing Bevy apps.

- [**`basic_builder`**](https://github.com/databasedav/jonmo/blob/main/examples/basic_builder.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/basic_builder/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/basic_builder/)

    a simple increasing timer using the entity builder, showcasing the recommended, idiomatic way to use jonmo signals

- [**`counter`**](https://github.com/databasedav/jonmo/blob/main/examples/counter.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/counter/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/counter/)

    the example above, a simple counter

- [**`lifetimes`**](https://github.com/databasedav/jonmo/blob/main/examples/lifetimes.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/lifetimes/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/lifetimes/)

    a reactive enumerated list of colors, each with an independent lifetime timer

- [**`letters`**](https://github.com/databasedav/jonmo/blob/main/examples/letters.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/letters/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/letters/)

    simple key press counter, showcasing map reactivity

- [**`filters`**](https://github.com/databasedav/jonmo/blob/main/examples/filters.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/filters/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/filters/)

    diverse filtering options for a list of items, showcasing vector reactivity

## Bevy compatibility

|bevy|jonmo|
|-|-|
|0.16|0.3|
|0.15|0.1|

## license
All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](https://github.com/databasedav/jonmo/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/databasedav/jonmo/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.

### your contributions
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
