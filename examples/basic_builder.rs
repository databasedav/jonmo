//! A simple increasing timer using the entity builder, showcasing the recommended, idiomatic way
//! to use jonmo signals.
mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(examples_plugin)
        .add_systems(
            Startup,
            (
                |world: &mut World| {
                    ui().spawn(world);
                },
                camera,
            ),
        )
        .add_systems(Update, incr_value)
        .insert_resource(ValueTicker(Timer::from_seconds(1., TimerMode::Repeating)))
        .run();
}

#[derive(Resource, Deref, DerefMut)]
struct ValueTicker(Timer);

#[derive(Component, Clone, Default, PartialEq, Deref)]
struct Value(i32);

fn ui() -> jonmo::Builder {
    jonmo::Builder::from(Node {
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        height: Val::Percent(100.),
        width: Val::Percent(100.),
        ..default()
    })
    .child({
        let lazy_entity = LazyEntity::new();
        jonmo::Builder::from((Node::default(), TextFont::from_font_size(100.)))
            .lazy_entity(lazy_entity.clone())
            .insert(Value(0))
            .component_signal(
                signal::from_component_changed::<Value>(lazy_entity)
                    .map_in(deref_copied)
                    .map_in_ref(ToString::to_string)
                    .map_in(Text)
                    .map_in(Some),
            )
    })
}

fn incr_value(mut ticker: ResMut<ValueTicker>, time: Res<Time>, mut values: Query<&mut Value>) {
    if ticker.tick(time.delta()).is_finished() {
        for mut value in values.iter_mut() {
            value.0 = value.0.wrapping_add(1);
        }
        ticker.reset();
    }
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
