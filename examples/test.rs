#![allow(missing_docs)]
mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    let numbers = MutableVec::builder().values([1, 2, 3, 4, 5]).spawn(app.world_mut());
    app.add_plugins(examples_plugin)
        .insert_resource(Numbers(numbers.clone()))
        .add_systems(
            PostStartup,
            (
                move |world: &mut World| {
                    ui_root(numbers.clone()).spawn(world);
                },
                camera,
            ),
        )
        .add_systems(Update, (live.run_if(any_with_component::<Lifetime>), hotkeys, toggle))
        .insert_resource(ToggleFilter(true))
        .run();
}

#[derive(Resource, Clone)]
struct Numbers(MutableVec<i32>);

#[derive(Component, Default, Clone)]
struct Lifetime(f32);

fn ui_root(numbers: MutableVec<i32>) -> jonmo::Builder {
    jonmo::Builder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(10.0),
        ..default()
    })
    .children_signal_vec(
        numbers
            .signal_vec()
            .filter_signal(|In(n)| {
                signal::from_system(move |In(_), toggle: Res<ToggleFilter>| n % 2 == if toggle.0 { 0 } else { 1 })
                    .cache()
            })
            .map(|In(n)| item(n)),
    )
}

#[derive(Resource)]
struct ToggleFilter(bool);

fn item(number: i32) -> jonmo::Builder {
    jonmo::Builder::from((
        Node {
            height: Val::Px(40.0),
            width: Val::Px(200.0),
            padding: UiRect::all(Val::Px(5.0)),
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::WHITE),
    ))
    .child(jonmo::Builder::from((
        Node {
            height: Val::Percent(100.),
            width: Val::Percent(100.),
            ..default()
        },
        Text(number.to_string()),
        TextColor(Color::BLACK),
        TextLayout::justify(Justify::Center),
        Lifetime::default(),
    )))
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn live(mut lifetimes: Query<&mut Lifetime>, time: Res<Time>) {
    for mut lifetime in lifetimes.iter_mut() {
        lifetime.0 += time.delta_secs();
    }
}

fn hotkeys(
    keys: Res<ButtonInput<KeyCode>>,
    numbers: ResMut<Numbers>,
    mut mutable_vec_datas: Query<&mut MutableVecData<i32>>,
) {
    let mut guard = numbers.0.write(&mut mutable_vec_datas);
    if keys.just_pressed(KeyCode::Equal) {
        guard.push((guard.len() + 1) as i32);
    } else if keys.just_pressed(KeyCode::Minus) {
        guard.pop();
    }
}

fn toggle(keys: Res<ButtonInput<KeyCode>>, mut toggle: ResMut<ToggleFilter>) {
    if keys.just_pressed(KeyCode::Space) {
        toggle.0 = !toggle.0;
    }
}
