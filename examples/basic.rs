//! A simple increasing timer without using the entity builder, showcasing the least invasive way
//! to start using jonmo signals in existing Bevy apps.

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
    let signal = signal::from_component_changed::<Value>(text)
        .map(move |In(value): In<Value>, mut commands: Commands| {
            commands.entity(text).insert(Text(value.0.to_string()));
        })
        .register(world);
    let mut ui_root = world.spawn(Node {
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        height: Val::Percent(100.),
        width: Val::Percent(100.),
        ..default()
    });
    ui_root.add_child(text);
    ui_root.insert(SignalHandles::from([signal]));
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
