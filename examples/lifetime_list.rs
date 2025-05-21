mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::{prelude::*, utils::LazyEntity};

fn main() {
    let mut app = App::new();
    let colors = MutableVec::from([random_color(), random_color()]);
    app.add_plugins(DefaultPlugins)
        .add_plugins(JonmoPlugin)
        .insert_resource(Colors(colors.clone()))
        .add_systems(
            PostStartup,
            (
                move |world: &mut World| {
                    ui_root(colors.signal_vec()).spawn(world);
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
struct Colors(MutableVec<Color>);

#[derive(Component, Default, Clone, Reflect)]
struct Lifetime(f32);

fn ui_root(colors: impl SignalVec<Item = Color>) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(10.0),
        ..default()
    })
    .children_signal_vec(
        colors
            .enumerate()
            .map(|In((index, color))| item(index, color)),
    )
}

fn item(index: impl Signal<Item = Option<usize>>, color: Color) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            height: Val::Px(40.0),
            width: Val::Px(250.0),
            padding: UiRect::all(Val::Px(5.0)),
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(color),
    ))
    .child({
        let parent = LazyEntity::new();
        JonmoBuilder::from((
            Node {
                height: Val::Percent(100.),
                width: Val::Percent(100.),
                ..default()
            },
            TextColor(Color::BLACK),
            TextLayout::new_with_justify(JustifyText::Center),
            Lifetime::default(),
        ))
        .entity_sync(parent.clone())
        .child(
            JonmoBuilder::from((
                Node {
                    height: Val::Percent(100.),
                    width: Val::Percent(100.),
                    ..default()
                },
                TextColor(Color::BLACK),
                TextLayout::new_with_justify(JustifyText::Center),
            ))
            .component_signal(
                index
                    .map(|In(index): In<Option<usize>>| format!("item {}", index.unwrap_or(0)))
                    .map(|In(text): In<String>| Text::new(text)),
            ),
        )
        .child((
            TextColor(Color::BLACK),
            TextLayout::new_with_justify(JustifyText::Center),
            Text::new(" | "),
        ))
        .child(
            JonmoBuilder::from((
                Node {
                    height: Val::Percent(100.),
                    width: Val::Percent(100.),
                    ..default()
                },
                TextColor(Color::BLACK),
                TextLayout::new_with_justify(JustifyText::Center),
            ))
            .component_signal(
                SignalBuilder::from_component_lazy(parent)
                    .map(|In(Lifetime(lifetime))| lifetime.round())
                    .dedupe()
                    .map(|In(lifetime): In<f32>| Text::new(format!("lifetime: {}", lifetime))),
            ),
        )
    })
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn live(mut lifetimes: Query<&mut Lifetime>, time: Res<Time>) {
    for mut lifetime in lifetimes.iter_mut() {
        lifetime.0 += time.delta_secs();
    }
}

fn hotkeys(keys: Res<ButtonInput<KeyCode>>, colors: ResMut<Colors>, mut commands: Commands) {
    let mut flush = false;
    if keys.just_pressed(KeyCode::Equal) {
        colors.0.push(random_color());
        flush = true;
    } else if keys.just_pressed(KeyCode::Minus) {
        colors.0.pop();
        flush = true;
    }
    if flush {
        commands.queue(colors.0.flush());
    }
}
