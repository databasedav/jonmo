//! Simple counter example, ported from haalka.

mod utils;
use utils::*;

use std::ops::Deref;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    App::new()
        .add_plugins(examples_plugin)
        .add_systems(
            Startup,
            (
                // The `JonmoBuilder` returned by `ui_root` is spawned into the world.
                |world: &mut World| {
                    ui_root().spawn(world);
                },
                camera,
            ),
        )
        .run();
}

/// A component to hold the state of our counter.
/// In jonmo, reactive signals are typically derived from component or resource data.
#[derive(Component, Clone, PartialEq, Debug, Deref)]
struct Counter(i32);

/// Constructs the root UI element for the counter.
fn ui_root() -> JonmoBuilder {
    // A LazyEntity acts as a placeholder for an entity that will be spawned later.
    // We'll use this to let the buttons and text refer to the entity that holds the
    // `Counter` component before it exists.
    let counter_holder = LazyEntity::new();

    JonmoBuilder::from(Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    })
    .child(
        // This Node will be a row and will also hold our state component.
        JonmoBuilder::from(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(15.0),
            align_items: AlignItems::Center, // Vertically center items in the row
            ..default()
        })
        // Insert the state component. This entity is now the source of truth.
        .insert(Counter(0))
        // Fulfill the LazyEntity promise. From now on, `counter_holder.get()` will
        // return the entity ID of this row.
        .entity_sync(counter_holder.clone())
        // Add the children: decrement button, text, increment button.
        .child(counter_button(counter_holder.clone(), "-", -1))
        .child(
            // This JonmoBuilder creates the text display.
            JonmoBuilder::from((Node::default(), TextFont::from_font_size(25.)))
                // `component_signal` is the core of jonmo's reactivity. It takes a signal
                // and uses its output to insert or update a component on this entity.
                .component_signal(
                    // Create a signal that reads the `Counter` component from our state holder.
                    SignalBuilder::from_component_lazy::<Counter>(counter_holder.clone())
                        // Map the `Counter` component to a `Text` component.
                        .map_in(|counter: Counter| *counter) // TODO: .map_in(Deref::deref) produces weird compiler errors but would be cool
                        .dedupe()
                        .map_in_ref(ToString::to_string)
                        .map_in(Text)
                        .map_in(Some),
                ),
        )
        .child(counter_button(counter_holder, "+", 1)),
    )
}

/// Creates a button that modifies the counter.
fn counter_button(counter_holder: LazyEntity, label: &'static str, step: i32) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            width: Val::Px(45.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BorderRadius::MAX,
        BackgroundColor(Color::hsl(300., 0.75, 0.75))
    ))
    // Insert Bevy's `Interaction` component so it can be tracked for clicks and hovers.
    // event listeners (observers).
    .on_spawn(move |world, entity| {
        // Observe changes to this entity's `Interaction` component to detect a click.
        world.entity_mut(entity).observe(clone!((counter_holder) move |
            click: Trigger<Pointer<Click>>,
            mut counters: Query<&mut Counter>
        | {
            if matches!(click.button, PointerButton::Primary) && let Ok(mut counter) = counters.get_mut(counter_holder.get()) {
                counter.0 += step;
            }

        }));
    })
    .child(
        // The button's label.
        JonmoBuilder::from((Text(label.to_string()), TextFont::from_font_size(25.))),
    )
}

/// Spawns the 2D camera.
fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
