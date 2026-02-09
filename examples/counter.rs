//! Simple counter example, ported from a similar example in haalka.
//!
//! This example introduces some concepts for building reactive Bevy applications using jonmo's
//! signals:
//!
//! 1. **World-Driven State**: The application's state (the counter's value) is stored in a standard
//!    Bevy `Component`, the "single source of truth".
//!
//! 2. **Declarative UI with `jonmo::Builder`**: The entire entity hierarchy is defined up-front in
//!    a clean, colocated, declarative style.
//!
//! 3. **The `LazyEntity` Pattern**: related entities can refer to the state-holding entity *before*
//!    it has been spawned, solving a common ordering problem in hierarchical entity construction.
//!
//! 4. **Reactivity with `.signal` methods**: A signal is created that reads the state component,
//!    whose output is then used to reactively update other components, like the `Text` for the
//!    counter's display.

mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    App::new()
        .add_plugins(examples_plugin)
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

fn ui_root() -> jonmo::Builder {
    // We need the buttons and the text display to know the `Entity` ID of the node
    // that will hold the `Counter` component. However, that node hasn't been spawned yet.
    // `LazyEntity` acts as a placeholder or a "promise" for an entity that will be
    // spawned by a `jonmo::Builder` later. It can be cloned and passed around freely, e.g. into the
    // bodies of signal systems.
    let counter_holder = LazyEntity::new();

    jonmo::Builder::from(Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    })
    .child(
        jonmo::Builder::from(Node {
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(15.0),
            align_items: AlignItems::Center,
            padding: UiRect::all(Val::Px(25.)),
            ..default()
        })
        .insert(Counter(0))
        // `.lazy_entity()` connects the `LazyEntity` to the actual entity that this `jonmo::Builder` spawns. Now
        // calling `*counter_holder` from other deferred contexts, e.g. in the bodies of signal systems,
        // will return the `Entity` ID of this row node.
        .lazy_entity(counter_holder.clone())
        .child(counter_button(counter_holder.clone(), PINK, "-", -1))
        .child(
            jonmo::Builder::from((Node::default(), TextFont::from_font_size(25.)))
                // `component_signal` is a core fixture of jonmo's reactivity. It takes a signal as an argument and
                // uses its output to insert or update a component on the entity being built. Here,
                // we're creating a signal that will produce a `Text` component whenever the counter
                // changes.
                .component_signal(
                    // `signal::from_component_changed` creates a signal that reactively reads a component from an
                    // entity that doesn't exist yet, identified by our `LazyEntity`, only firing when the component
                    // changes.
                    signal::from_component_changed::<Counter>(counter_holder.clone())
                        // `map_in` is a shorthand for `.map` that takes a regular function instead of a Bevy
                        // system; this is especially convenient when additional `SystemParam`s aren't necessary.
                        // `deref_copied` dereferences and copies, extracting the inner `i32` from the `Counter`
                        // newtype for further transformation.
                        .map_in(deref_copied)
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

fn counter_button(counter_holder: LazyEntity, color: Color, label: &'static str, step: i32) -> jonmo::Builder {
    jonmo::Builder::from((
        Node {
            width: Val::Px(45.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BorderRadius::MAX,
        BackgroundColor(color),
    ))
    // Attach observers to the entity
    .observe(move |_: On<Pointer<Click>>, mut counters: Query<&mut Counter>| {
        // Use the fulfilled `LazyEntity` to get mutable access to the `Counter` component on our
        // state-holding entity.
        **counters.get_mut(*counter_holder).unwrap() += step;
        // Because our text display has a signal that reads this component, this change will
        // automatically trigger a UI update at the end of the frame.
    })
    .child(jonmo::Builder::from((Text::from(label), TextFont::from_font_size(25.))))
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
