mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    let numbers = MutableVec::from([1, 2, 3, 4, 5]);
    app.add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin)
        .insert_resource(Numbers(numbers.clone()))
        .add_systems(
            PostStartup,
            (
                move |world: &mut World| {
                    ui_root(numbers.signal_vec()).spawn(world);
                },
                camera,
            ),
        )
        .add_systems(
            Update,
            (live.run_if(any_with_component::<Lifetime>), hotkeys),
        )
        .run();
}

#[derive(Resource, Clone)]
struct Numbers(MutableVec<u32>);

#[derive(Component, Default, Clone, Reflect)]
struct Lifetime(f32);

fn ui_root(numbers: impl SignalVec<Item = u32>) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(10.0),
        ..default()
    })
    .children_signal_vec(numbers.filter_map(|In(i)| if i % 2 == 0 { Some(i * 10) } else { None }).map(|In(color)| item(color)))
}

fn item(number: u32) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            height: Val::Px(40.0),
            width: Val::Px(200.0),
            padding: UiRect::all(Val::Px(5.0)),
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::WHITE),
    ))
    .child(
        JonmoBuilder::from((
            Node {
                height: Val::Percent(100.),
                width: Val::Percent(100.),
                ..default()
            },
            Text(number.to_string()),
            TextColor(Color::BLACK),
            TextLayout::new_with_justify(JustifyText::Center),
            Lifetime::default(),
        ))
    )
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn live(mut lifetimes: Query<&mut Lifetime>, time: Res<Time>) {
    for mut lifetime in lifetimes.iter_mut() {
        lifetime.0 += time.delta_secs();
    }
}

fn hotkeys(keys: Res<ButtonInput<KeyCode>>, numbers: ResMut<Numbers>, mut commands: Commands) {
    let mut flush = false;
    if keys.just_pressed(KeyCode::Equal) {
        numbers.0.push((numbers.0.len() + 1) as u32);
        flush = true;
    } else if keys.just_pressed(KeyCode::Minus) {
        numbers.0.pop();
        flush = true;
    }
    if flush {
        commands.queue(numbers.0.flush());
    }
}
