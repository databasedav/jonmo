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

    app.add_plugins(examples_plugin)
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
    .child(
        JonmoBuilder::from((
            Node {
                height: Val::Px(40.),
                width: Val::Px(100.),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(bevy::color::palettes::basic::GREEN.into()),
        ))
        // `on_spawn` runs a closure with access to the `World` and the spawned `Entity`
        // just after the entity is created. This is a good place to set up observers or
        // other one-time logic.
        .on_spawn(|world, entity| {
            // `observe` is a Bevy event-handling pattern. Here, we're setting up this
            // button entity to listen for a `Click` event.
            world.entity_mut(entity).observe(
                // This closure is the event handler that runs when the button is clicked.
                move |_: Trigger<Pointer<Click>>, colors: Res<Colors>, mut commands: Commands| {
                    // Try to get the `Index` component from the clicked entity.
                    // We found the index! Now we can mutate the central data source.
                    // `colors.0.write()` gets a write lock on the `MutableVec`.
                    let mut guard = colors.0.write();
                    guard.insert(guard.len(), random_color());
                    // IMPORTANT: After mutating a `MutableVec`, you must call `flush()`
                    // to broadcast the changes to all listening signals.
                    // We queue the flush command to be run at the end of the frame.
                    commands.queue(colors.0.flush());
                },
            );
        })
        .child(JonmoBuilder::from((
            Node::default(),
            Text::new("+"),
            TextColor(Color::WHITE),
            TextLayout::new_with_justify(JustifyText::Center),
        ))),
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
                        .map_in(|index| index.as_ref().map(ToString::to_string).map(TextSpan)),
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
                        .map_in(TextSpan)
                        .map_in(Some),
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
        .component_signal(index.map_in(|index| index.map(Index)))
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
