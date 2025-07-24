//! Simple counter example, ported from a similar example in the `haalka` UI library.
//!
//! This example demonstrates the fundamental concepts of `jonmo` for building
//! interactive UIs:
//!
//! 1. **Component-Driven State**: The application's state (the counter's value) is stored in a
//!    standard Bevy `Component`. This is the "single source of truth".
//!
//! 2. **Declarative UI with `JonmoBuilder`**: The entire UI tree is defined up-front in a clean,
//!    declarative style.
//!
//! 3. **The `LazyEntity` Pattern**: UI elements (like buttons and text) can refer to the
//!    state-holding entity *before* it has been spawned, solving a common ordering problem in UI
//!    construction.
//!
//! 4. **Reactivity with `component_signal`**: A signal is created that reads the state component.
//!    Its output is then used to reactively update other components, like the `Text` for the
//!    counter's display.
//!
//! 5. **Event Handling with `observe`**: User input (a button click) is handled by a Bevy observer
//!    system that mutates the state component, which in turn triggers the reactive signal chain.

// The `utils` module contains boilerplate for setting up a basic Bevy app.
mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    App::new()
        // The `examples_plugin` includes `DefaultPlugins` and the `JonmoPlugin`.
        // `JonmoPlugin` adds the necessary systems for the reactive graph to run.
        .add_plugins(examples_plugin)
        .add_systems(
            Startup,
            (
                // This is a common pattern for spawning a `JonmoBuilder` at startup.
                // The closure takes a mutable world reference, calls our UI-defining
                // function (`ui_root`), and then spawns the resulting builder.
                |world: &mut World| {
                    ui_root().spawn(world);
                },
                // A standard Bevy system to spawn a camera is also needed.
                camera,
            ),
        )
        .run();
}

/// A component to hold the state of our counter.
///
/// In `jonmo`, the "source of truth" for reactive data is typically a standard Bevy
/// `Component` or `Resource`. Signals are then derived from this data.
#[derive(Component, Clone, Deref, DerefMut)]
struct Counter(i32);

/// Constructs the root `JonmoBuilder` for the entire counter UI.
fn ui_root() -> JonmoBuilder {
    // --- The `LazyEntity` Pattern ---
    // We need the buttons and the text display to know the `Entity` ID of the node
    // that will hold the `Counter` component. However, that node hasn't been spawned yet.
    // `LazyEntity` acts as a placeholder or a "promise" for an entity that will be
    // spawned by a `JonmoBuilder` later. It can be cloned and passed around freely.
    let counter_holder = LazyEntity::new();

    // The root of our UI is a full-screen node that centers its content.
    JonmoBuilder::from(Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    })
    .child(
        // This `Node` will be a horizontal row for our UI elements.
        // It's also the entity we've chosen to hold the counter's state.
        JonmoBuilder::from(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(15.0),
            align_items: AlignItems::Center, // Vertically center items in the row
            ..default()
        })
        // --- State Declaration ---
        // We insert the `Counter` component here. This entity is now the single
        // source of truth for the counter's value.
        .insert(Counter(0))
        // --- Fulfilling the Promise ---
        // `.entity_sync()` connects the `LazyEntity` to the actual entity that this
        // `JonmoBuilder` spawns. From this point on, calling `counter_holder.get()`
        // will return the `Entity` ID of this row node.
        .entity_sync(counter_holder.clone())
        // Now that the `counter_holder` promise is set to be fulfilled, we can
        // create the child elements that depend on it.
        .child(counter_button(counter_holder.clone(), "-", -1))
        .child(
            // This `JonmoBuilder` creates the text display for the counter value.
            JonmoBuilder::from((Node::default(), TextFont::from_font_size(25.)))
                // --- Reactivity ---
                // `component_signal` is the heart of `jonmo`'s reactivity. It takes a
                // signal as an argument and uses its output to insert or update a
                // component on the entity being built. Here, we're creating a signal
                // that will produce a `Text` component whenever the counter changes.
                .component_signal(
                    // `SignalBuilder::from_component_lazy` creates a signal that reactively
                    // reads a component from an entity that doesn't exist yet, identified
                    // by our `LazyEntity`.
                    SignalBuilder::from_component_lazy::<Counter>(counter_holder.clone())
                        // The signal chain transforms the data step-by-step:
                        // 1. The signal emits `Counter(i32)`. `.map_in` extracts the inner `i32`.
                        .map_in(|counter: Counter| *counter)
                        // 2. `.dedupe()` is a crucial optimization. It ensures the rest of the chain only runs when the
                        //    counter's value *actually changes*, preventing redundant updates every frame.
                        .dedupe()
                        // 3. Convert the `i32` into a `String`.
                        .map_in_ref(ToString::to_string)
                        // 4. Wrap the `String` in Bevy's `Text` component.
                        .map_in(Text)
                        // 5. `component_signal` expects an `Option<Component>`. If the signal produces `None`, the
                        //    component is removed. Here, our value is always present, so we wrap it in `Some`.
                        .map_in(Some),
                ),
        )
        .child(counter_button(counter_holder, "+", 1)),
    )
}

/// A factory function that creates a `JonmoBuilder` for a counter button.
///
/// # Arguments
/// * `counter_holder`: A `LazyEntity` pointing to the node with the `Counter` component.
/// * `label`: The text to display on the button (e.g., "+" or "-").
/// * `step`: The amount to add to the counter when this button is clicked.
fn counter_button(counter_holder: LazyEntity, label: &'static str, step: i32) -> JonmoBuilder {
    JonmoBuilder::from((
        // A styled node for the button's appearance.
        Node {
            width: Val::Px(45.0),
            height: Val::Px(45.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BorderRadius::MAX,
        BackgroundColor(Color::hsl(300., 0.75, 0.75)),
    ))
    // The button needs to react to user input. We achieve this by setting up an
    // event listener when the button entity is spawned.
    //
    // `.on_spawn()` provides a hook to run code on the `World` and the button's `Entity`
    // right after it has been created. This is the ideal place to set up listeners.
    .on_spawn(move |world, entity| {
        // --- Event Handling ---
        // We set up an `observe` system on the button's entity. This is a Bevy pattern
        // for running a system in response to a specific event on a specific entity.
        // Bevy's UI plugins automatically handle hit-testing and generating Pointer events
        // for UI nodes.
        world.entity_mut(entity).observe(
            move |// The `Trigger` contains information about the event that fired.
                  // Here, we listen for `Pointer<Click>`.
                  trigger: Trigger<Pointer<Click>>,
                  // We request write access to all `Counter` components in the world.
                  mut counters: Query<&mut Counter>| {
                // Check if it was a primary (e.g., left) mouse click.
                if matches!(trigger.button, PointerButton::Primary) {
                    // Use the fulfilled `LazyEntity` to get mutable access to the *specific*
                    // `Counter` component on our state-holding entity.
                    if let Ok(mut counter) = counters.get_mut(counter_holder.get()) {
                        // --- State Mutation ---
                        // This is the line that changes the application's state.
                        // Because our text display has a signal that reads this component,
                        // this change will automatically trigger a UI update on the next
                        // run of the `jonmo` plugin's systems.
                        **counter += step;
                    }
                }
            },
        );
    })
    .child(
        // The button's text label.
        JonmoBuilder::from((Text::from(label), TextFont::from_font_size(25.))),
    )
}

/// Spawns the 2D camera, which is required for UI to be visible.
fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
