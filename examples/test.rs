#![allow(missing_docs)]
#![allow(unused_variables)]
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
                    ui_root(numbers.clone(), world).spawn(world);
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

#[rustfmt::skip]
fn ui_root(numbers: MutableVec<i32>, world: &mut World) -> jonmo::Builder {
    let list_a = MutableVec::builder().values([1, 2, 3, 4, 5]).spawn(world).signal_vec();
    let list_b = MutableVec::builder().values([3, 4, 5]).spawn(world).signal_vec();
    let map = MutableBTreeMap::builder().values([(1, 2), (2, 3)]).spawn(world);
    // map.signal_map().
    jonmo::Builder::from(Node {
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
        // MutableVec::from([numbers.signal_vec(), numbers.signal_vec()]).signal_vec()
        numbers
            .signal_vec()
            // signal::from_system(|_: In<()>| 1)
            // signal::from_system(|_: In<()>, toggle: Res<ToggleFilter>| toggle.0)
            // .dedupe()
            // .switch_signal_vec(move |In(toggle)| {
            //     if toggle {
            //         // println!("toggled to A");
            //         list_a.clone()
            //     } else {
            //         // println!("toggled to B");
            //         list_b.clone()
            //     }
            // })
            // .dedupe()
            // .to_signal_vec()
            .filter_signal(|In(n)| {
                signal::from_system(move |_: In<()>, toggle: Res<ToggleFilter>| {
                    n % 2 == if toggle.0 { 0 } else { 1 }
                })
            })
            // .map_signal(|In(n): In<i32>| {
            //     signal::from_system(move |_: In<()>| n + 1).dedupe()
            // })
            // .debug()
            // .map_in(|n: i32| -n)
            // .sort_by_cmp()
            // .flatten()
            // .chain(numbers)
            // .intersperse(0)
            // .sort_by(|In((left, right)): In<(i32, i32)>| left.cmp(&right).reverse())
            // .sort_by_key(|In(n): In<i32>| -n)
            // .intersperse_with(|In(index_signal): In<jonmo::signal::Dedupe<jonmo::signal::Source<Option<usize>>>>, world: &mut World| {
            //     let signal = index_signal.debug().register(world);
            //     poll_signal(world, *signal)
            //         .and_then(downcast_any_clone::<Option<usize>>).flatten().unwrap_or_default() as i32
            // })
            .map(|In(n)| item(n))
            // .intersperse_with(
            //     |index_signal: In<jonmo::signal::Dedupe<jonmo::signal::Source<Option<usize>>>>| {
            //         jonmo::Builder::from(Node::default()).component_signal(
            //             index_signal
            //                 .debug()
            //                 .map_in(|idx_opt| Text::new(format!("{}", idx_opt.unwrap_or(0)))),
            //         )
            //     },
            // ),
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
        TextLayout::new_with_justify(Justify::Center),
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
