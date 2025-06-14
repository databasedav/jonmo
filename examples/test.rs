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
                    ui_root(numbers.clone()).spawn(world);
                },
                camera,
            ),
        )
        .add_systems(
            Update,
            (live.run_if(any_with_component::<Lifetime>), hotkeys, toggle),
        )
        .insert_resource(ToggleFilter(true))
        .run();
}

#[derive(Resource, Clone)]
struct Numbers(MutableVec<i32>);

#[derive(Component, Default, Clone, Reflect)]
struct Lifetime(f32);

fn ui_root(numbers: MutableVec<i32>) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(10.0),
        ..default()
    })
    // .child_signal(numbers.clone().is_empty().map(|In(len)| item(len as u32)))
    .children_signal_vec(
        MutableVec::from([numbers.signal_vec(), numbers.signal_vec()]).signal_vec()
        // numbers.clone()
            // .filter_signal(|In(n)| {
            //     SignalBuilder::from_system(
            //         move |_: In<()>, toggle: Res<ToggleFilter>| {
            //             n % 2 == if toggle.0 { 0 } else { 1 }
            //         },
            //     )
            // })
            // .map_signal(|In(n): In<i32>| {
            //     SignalBuilder::from_system(move |_: In<_>| n + 1)
            // })
            // .chain(numbers)
            // .intersperse(0)
            // .intersperse_with(|_: In<_>| 0)
            // .sort_by(|In((left, right)): In<(i32, i32)>| left.cmp(&right).reverse())
            // .sort_by_key(|In(n): In<i32>| -n)
            .flatten()
            .map(|In(n)| item(n)),
    )
}

#[derive(Resource)]
struct ToggleFilter(bool);

fn item(number: i32) -> JonmoBuilder {
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
    .child(JonmoBuilder::from((
        Node {
            height: Val::Percent(100.),
            width: Val::Percent(100.),
            ..default()
        },
        Text(number.to_string()),
        TextColor(Color::BLACK),
        TextLayout::new_with_justify(JustifyText::Center),
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

fn hotkeys(keys: Res<ButtonInput<KeyCode>>, numbers: ResMut<Numbers>, mut commands: Commands) {
    let mut flush = false;
    let mut guard = numbers.0.write();
    if keys.just_pressed(KeyCode::Equal) {
        guard.push((guard.len() + 1) as i32);
        flush = true;
    } else if keys.just_pressed(KeyCode::Minus) {
        guard.pop();
        flush = true;
    }
    if flush {
        commands.queue(numbers.0.flush());
    }
}

fn toggle(keys: Res<ButtonInput<KeyCode>>, mut toggle: ResMut<ToggleFilter>) {
    if keys.just_pressed(KeyCode::Space) {
        toggle.0 = !toggle.0;
    }
}
