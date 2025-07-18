# jonmo [জন্ম](https://translate.google.com/?sl=bn&tl=en&text=%E0%A6%9C%E0%A6%A8%E0%A7%8D%E0%A6%AE&op=translate)

[![Crates.io Version](https://img.shields.io/crates/v/jonmo?style=for-the-badge)](https://crates.io/crates/jonmo)
[![Docs.rs](https://img.shields.io/docsrs/jonmo?style=for-the-badge)](https://docs.rs/jonmo)
[![Following released Bevy versions](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue?style=for-the-badge)](https://bevyengine.org/learn/quick-start/plugin-development/#main-branch-tracking)

```text
in bengali, jonmo means "birth"
```

[jonmo](https://github.com/databasedav/jonmo) provides an ergonomic, functional, and declarative API for specifying Bevy [system](https://docs.rs/bevy/latest/bevy/ecs/system/index.html) dependency graphs, where "output" handles to nodes of the graph are canonically referred to as "signals". Building upon these signals, jonmo offers a high level [entity builder](https://docs.rs/jonmo/latest/jonmo/builder/struct.JonmoBuilder.html) which enables one to declare reactive entities, components, and children using a familiar fluent syntax with semantics and API ported from the incredible [FRP](https://en.wikipedia.org/wiki/Functional_reactive_programming) signals of [futures-signals](https://github.com/Pauan/rust-signals) and its web UI dependents [MoonZoon](https://github.com/MoonZoon/MoonZoon) and [Dominator](https://github.com/Pauan/rust-dominator).

The runtime of jonmo is quite simple; every frame, the outputs of systems are forwarded to their dependants, recursively. The complexity and power of jonmo really emerges from its monadic signal combinators, defined within the [`SignalExt`](https://docs.rs/jonmo/latest/jonmo/signal/trait.SignalExt.html), [`SignalVecExt`](https://docs.rs/jonmo/latest/jonmo/signal_vec/trait.SignalVecExt.html), and [`SignalMapExt`](https://docs.rs/jonmo/latest/jonmo/signal_map/trait.SignalMapExt.html) traits (ported from futures-signals' traits of the same name), which internally manage special Bevy systems that allow for the declarative composition of complex data flows with minimalistic, high-level, signals-oriented methods.

### Assorted features:
- fine-grained reactivity for all entities, components, and children
- ***diff-less*** constant-time reactive updates for collections (available through `MutableVec` and `MutableBTreeMap`)
- automated system lifecycle management when using the builder API, simple component on-remove hook when not
- polling API for when one needs an escape hatch from the regular push-based output semantics (polling is used sparsely internally for some combinators)
- either wrappers (a la https://github.com/rayon-rs/either) and type-erased signals (via boxing) for cheap and flexible management of distinct signal types from different branches of logic
- `no_std` *always*

## examples
```rust no_run
//! This example showcases a more advanced, dynamic UI using `jonmo`.
//! It demonstrates:
//! - A reactive list of items whose length can change at runtime.
//! - How each item in the list can have its own internal state and reactive updates.
//! - Communication from a child UI element (a "remove" button) back to the central data source.
//! - The use of `LazyEntity` to create signals that depend on an entity that hasn't been spawned
//!   yet.
//!
//! The application displays a list of colored bars. Each bar has a "lifetime" counter that
//! continuously updates. You can add new bars by pressing the `=` key, remove the last bar with
//! the `-` key, or click the red 'x' button on any bar to remove it specifically.
//! This pattern is fundamental for building complex, data-driven UIs like settings menus,
//! inventory screens, or leaderboards.

// This example uses a few helper functions, like `random_color`.
mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();

    // 1. --- DATA SOURCE SETUP ---
    // `MutableVec` is the core reactive data source for lists in `jonmo`.
    // We initialize it with two random colors.
    // It's wrapped in an `Arc<RwLock<...>>` internally, so cloning it is cheap
    // and allows multiple systems to access and modify the same data.
    let colors = MutableVec::from([random_color(), random_color()]);

    app.add_plugins(DefaultPlugins)
        // Add the main `jonmo` plugin which contains the systems for signal processing.
        .add_plugins(JonmoPlugin)
        // 2. --- RESOURCE MANAGEMENT ---
        // We insert a clone of our `MutableVec` into a Bevy resource. This makes it
        // accessible to any system that needs to read or modify the list of colors,
        // such as our `hotkeys` system or the remove button's `observe` system.
        .insert_resource(Colors(colors.clone()))
        .add_systems(
            // We use `PostStartup` to ensure that Bevy's UI systems are initialized
            // before we try to spawn our UI.
            Startup,
            (
                // 3. --- UI SPAWNING ---
                // We move the `colors` `MutableVec` into a closure that will spawn the UI.
                // `colors.signal_vec()` creates a `SignalVec`, which is a stream of
                // changes (`VecDiff`s) that other parts of the UI can subscribe to.
                move |world: &mut World| {
                    ui_root(colors.signal_vec()).spawn(world);
                },
                camera,
            ),
        )
        // 4. --- UPDATE SYSTEMS ---
        // These systems run every frame.
        .add_systems(
            Update,
            (
                // The `live` system increments the lifetime of each list item.
                // It only runs if there is at least one entity with a `Lifetime` component.
                live.run_if(any_with_component::<Lifetime>),
                // The `hotkeys` system listens for keyboard input to add/remove colors.
                hotkeys,
            ),
        )
        .run();
}

/// A Bevy resource that holds a clone of the `MutableVec` of colors.
/// This allows different systems to easily access the central data source.
#[derive(Resource, Clone)]
struct Colors(MutableVec<Color>);

/// A component to track the "lifetime" of a list item, in seconds.
/// We'll use this to demonstrate that each item in the reactive list
/// can have its own independent, stateful logic.
#[derive(Component, Default, Clone)]
struct Lifetime(f32);

/// Constructs the root UI node.
///
/// It takes a `SignalVec` of `Color`s as input. This is the reactive "pipe"
/// that will drive the creation, destruction, and updating of child elements.
fn ui_root(colors: impl SignalVec<Item = Color>) -> JonmoBuilder {
    // A standard vertical flexbox to hold our list items.
    JonmoBuilder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(10.0),
        ..default()
    })
    // This is the core of the dynamic list.
    // `children_signal_vec` subscribes to a `SignalVec`. For each item in the
    // vector, it spawns a child entity using the `JonmoBuilder` returned by the closure.
    // It handles all diffs automatically: `Push` creates a new child, `RemoveAt`
    // despawns one, `Move` reorders them, etc.
    .children_signal_vec(
        // `.enumerate()` is a powerful combinator that transforms a `SignalVec<T>`
        // into a `SignalVec<(Signal<Option<usize>>, T)>`.
        // The first element of the tuple is a *new signal* that will always contain
        // the current index of that specific item, or `None` if it has been removed.
        // This is crucial for displaying the index or for actions like removing a specific item.
        colors.enumerate().map_in(|(index, color)| item(index, color)),
    )
}

/// A component to hold the index of a list item. This is inserted onto the
/// "remove" button so that when it's clicked, we know which item in the
/// `MutableVec` to remove.
#[derive(Component, Clone)]
struct Index(usize);

/// Constructs a `JonmoBuilder` for a single item in our list.
///
/// # Arguments
/// * `index` - A `Signal<Item = Option<usize>>` that provides the current index of this item. This
///   signal is provided by the `.enumerate()` call in `ui_root`.
/// * `color` - The `Color` for this specific item.
fn item(index: impl Signal<Item = Option<usize>> + Clone, color: Color) -> JonmoBuilder {
    // --- The LazyEntity Pattern ---
    // `LazyEntity` is a thread-safe, clone-able handle to an `Entity` that can be
    // created *before* the entity is spawned.
    // We need this because we want to create a signal for the text display that *reads*
    // the `Lifetime` component from its own parent entity. When we define the text signal,
    // the parent entity doesn't exist yet. `LazyEntity` acts as a promise that will be
    // fulfilled later.
    let lifetime_holder = LazyEntity::new();

    JonmoBuilder::from((
        Node {
            height: Val::Px(40.0),
            width: Val::Px(350.0),
            align_items: AlignItems::Center,
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(10.0),
            ..default()
        },
        // Each item gets its own `Lifetime` component, which will be updated by the `live` system.
        Lifetime::default(),
    ))
    // Here we fulfill the promise. `entity_sync` will set the `Entity` id into the
    // `lifetime_holder` once this `JonmoBuilder` is spawned into an actual entity.
    // Any signals that were created using `lifetime_holder` will now point to the correct entity.
    .entity_sync(lifetime_holder.clone())
    .child({
        // The main info panel for the item.
        JonmoBuilder::from((
            Node {
                height: Val::Percent(100.),
                width: Val::Percent(90.),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(color),
        ))
        .child(
            // This `JonmoBuilder` will hold the text. Bevy UI text is composed of `TextSection`s,
            // which are children of a `Text` entity. We use `JonmoBuilder`'s child methods to
            // construct these sections reactively.
            JonmoBuilder::from((
                Node::default(),
                // Start with a `Text` component with no sections. We'll add them as children.
                Text::new(""),
                TextColor(Color::BLACK), // Default color, can be overridden by children.
                TextLayout::new_with_justify(JustifyText::Center),
            ))
            // Child 1: A static text span.
            .child((TextColor(Color::BLACK), TextSpan::new("item ")))
            // Child 2: A reactive text span for the index.
            .child(
                JonmoBuilder::from(TextColor(Color::BLACK)).component_signal(
                    // `component_signal` takes a signal and uses its output to insert/update a component.
                    index
                        .clone()
                        // The index signal is `Option<usize>`. `unwrap_or_default` handles the case where it might be
                        // `None`.
                        .map_in(Option::unwrap_or_default)
                        // Convert the `usize` to a `String`.
                        .map_in_ref(ToString::to_string)
                        // Wrap the `String` in a `TextSpan`, which is the component `component_signal` will manage.
                        .map_in(TextSpan),
                ),
            )
            // Child 3: Another static text span.
            .child((TextColor(Color::BLACK), TextSpan::new(" | lifetime: ")))
            // Child 4: A reactive text span for the lifetime.
            .child(
                JonmoBuilder::from(TextColor(Color::BLACK)).component_signal(
                    // This is where the `LazyEntity` becomes powerful.
                    // We create a signal that reads a component from the entity that `lifetime_holder` will eventually
                    // point to.
                    SignalBuilder::from_component_lazy(lifetime_holder)
                        // Map the `Lifetime` component to its inner `f32` value and round it.
                        .map_in(|Lifetime(lifetime)| lifetime.round())
                        // `dedupe` is a crucial optimization. It ensures the rest of the signal chain only runs
                        // when the rounded lifetime value actually changes (once per second in this case),
                        // not on every single frame.
                        .dedupe()
                        // Convert the rounded `f32` to a `String`.
                        .map_in_ref(ToString::to_string)
                        // Wrap it in a `TextSpan` component for display.
                        .map_in(TextSpan),
                ),
            ),
        )
    })
    // Add the "remove" button as a child of the item row.
    .child(
        JonmoBuilder::from((
            Node {
                height: Val::Percent(100.),
                width: Val::Percent(10.),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            // Using a color from Bevy's built-in palette for the button.
            BackgroundColor(bevy::color::palettes::basic::RED.into()),
        ))
        // `on_spawn` runs a closure with access to the `World` and the spawned `Entity`
        // just after the entity is created. This is a good place to set up observers or
        // other one-time logic.
        .on_spawn(|world, entity| {
            // `observe` is a Bevy event-handling pattern. Here, we're setting up this
            // button entity to listen for a `Click` event.
            world.entity_mut(entity).observe(
                // This closure is the event handler that runs when the button is clicked.
                move |_: Trigger<Pointer<Click>>,
                      indices: Query<&Index>,
                      colors: Res<Colors>,
                      mut commands: Commands| {
                    // Try to get the `Index` component from the clicked entity.
                    if let Ok(&Index(index)) = indices.get(entity) {
                        // We found the index! Now we can mutate the central data source.
                        // `colors.0.write()` gets a write lock on the `MutableVec`.
                        colors.0.write().remove(index);
                        // IMPORTANT: After mutating a `MutableVec`, you must call `flush()`
                        // to broadcast the changes to all listening signals.
                        // We queue the flush command to be run at the end of the frame.
                        commands.queue(colors.0.flush());
                    }
                },
            );
        })
        // To make the observer work, the button entity needs to *have* an `Index` component.
        // We use `component_signal` again to reactively insert the `Index` component,
        // driven by the same `index` signal we used for the display text.
        .component_signal(index.map_in(Option::unwrap_or_default).map_in(Index))
        .child(JonmoBuilder::from((
            Node::default(),
            Text::new("x"),
            TextColor(Color::WHITE),
            TextLayout::new_with_justify(JustifyText::Center),
        ))),
    )
}

/// A standard Bevy system to spawn a 2D camera.
fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// This system runs every frame and is responsible for updating the `Lifetime`
/// of every list item. This change is automatically picked up by the reactive
/// text signal we defined in the `item` builder.
fn live(mut lifetimes: Query<&mut Lifetime>, time: Res<Time>) {
    for mut lifetime in lifetimes.iter_mut() {
        lifetime.0 += time.delta_secs();
    }
}

/// This system handles keyboard input for adding and removing colors from the main
/// `MutableVec` data source.
fn hotkeys(keys: Res<ButtonInput<KeyCode>>, colors: ResMut<Colors>, mut commands: Commands) {
    let mut flush = false;
    // `colors.0.write()` acquires a write lock on the `MutableVec`. The lock is
    // released when `guard` goes out of scope.
    let mut guard = colors.0.write();
    if keys.just_pressed(KeyCode::Equal) {
        // Add a new random color to the end of the vector.
        guard.push(random_color());
        flush = true;
    } else if keys.just_pressed(KeyCode::Minus) {
        // Remove the last color from the vector.
        guard.pop();
        flush = true;
    }

    // If we made any changes, we must flush them. This signals to `children_signal_vec`
    // and any other listeners that the data has changed, so they can update accordingly.
    if flush {
        commands.queue(colors.0.flush());
    }
}
```

## Bevy compatibility

|bevy|jonmo|
|-|-|
|0.16|0.2|
|0.15|0.1|

## license
All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](https://github.com/databasedav/jonmo/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/databasedav/jonmo/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.

### your contributions
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
