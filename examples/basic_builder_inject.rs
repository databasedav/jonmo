//! A simple increasing timer, injecting the entity builder into an existing entity, showcasing a
//! less invasive way to start using jonmo signals in existing Bevy apps.
mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(examples_plugin)
        .add_systems(Startup, (ui, camera))
        .add_systems(Update, incr_value)
        .insert_resource(ValueTicker(Timer::from_seconds(1., TimerMode::Repeating)))
        .run();
}

#[derive(Resource, Deref, DerefMut)]
struct ValueTicker(Timer);

#[derive(Component, Clone, Default, PartialEq)]
struct Value(i32);

fn ui(world: &mut World) {
    let text = world
        .spawn((Node::default(), TextFont::from_font_size(100.), Value(0)))
        .id();
    let mut ui_root = world.spawn(Node {
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        height: Val::Percent(100.),
        width: Val::Percent(100.),
        ..default()
    });
    ui_root.add_child(text);

    JonmoBuilder::new()
        .component_signal_from_component(|signal| signal.dedupe().map(|In(value): In<Value>| Text(value.0.to_string())))
        .spawn_on_entity(world, text);
}

fn incr_value(mut ticker: ResMut<ValueTicker>, time: Res<Time>, mut values: Query<&mut Value>) {
    if ticker.tick(time.delta()).finished() {
        for mut value in values.iter_mut() {
            value.0 = value.0.wrapping_add(1);
        }
        ticker.reset();
    }
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
