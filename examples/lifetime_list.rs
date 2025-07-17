//!

mod utils;
use utils::*;

use bevy::prelude::*;
use jonmo::prelude::*;

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
        .add_systems(Update, (live.run_if(any_with_component::<Lifetime>), hotkeys))
        .run();
}

#[derive(Resource, Clone)]
struct Colors(MutableVec<Color>);

#[derive(Component, Default, Clone)]
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
    .children_signal_vec(colors.enumerate().map_in(|(index, color)| item(index, color)))
}

#[derive(Component, Clone)]
struct Index(usize);

fn item(index: impl Signal<Item = Option<usize>> + Clone, color: Color) -> JonmoBuilder {
    let lifetime_holder = LazyEntity::new();
    JonmoBuilder::from((
        Node {
            height: Val::Px(40.0),
            width: Val::Px(350.0),
            align_items: AlignItems::Center,
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(10.0),
            ..default()
        },
        Lifetime::default(),
    ))
    .entity_sync(lifetime_holder.clone())
    .child({
        JonmoBuilder::from((
            Node {
                height: Val::Percent(100.),
                width: Val::Percent(90.),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(color),
        ))
        .child(
            JonmoBuilder::from((
                Node::default(),
                Text::new(""),
                TextColor(Color::BLACK),
                TextLayout::new_with_justify(JustifyText::Center),
            ))
            .child((TextColor(Color::BLACK), TextSpan::new("item ")))
            .child(
                JonmoBuilder::from(TextColor(Color::BLACK)).component_signal(
                    index
                        .clone()
                        .map_in(Option::unwrap_or_default)
                        .map_in_ref(ToString::to_string)
                        .map_in(TextSpan),
                ),
            )
            .child((TextColor(Color::BLACK), TextSpan::new(" | lifetime: ")))
            .child(
                JonmoBuilder::from(TextColor(Color::BLACK)).component_signal(
                    SignalBuilder::from_component_lazy(lifetime_holder)
                        .map_in(|Lifetime(lifetime)| lifetime.round())
                        .dedupe()
                        .map_in_ref(ToString::to_string)
                        .map_in(TextSpan),
                ),
            ),
        )
    })
    .child(
        JonmoBuilder::from((
            Node {
                height: Val::Percent(100.),
                width: Val::Percent(10.),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(bevy::color::palettes::basic::RED.into()),
        ))
        .on_spawn(|world, entity| {
            world.entity_mut(entity).observe(
                move |_: Trigger<Pointer<Click>>,
                      indices: Query<&Index>,
                      colors: Res<Colors>,
                      mut commands: Commands| {
                    if let Ok(&Index(index)) = indices.get(entity) {
                        colors.0.write().remove(index);
                        commands.queue(colors.0.flush());
                    }
                },
            );
        })
        .component_signal(index.map_in(Option::unwrap_or_default).map_in(Index))
        .child(JonmoBuilder::from((
            Node::default(),
            Text::new("x"),
            TextColor(Color::WHITE),
            TextLayout::new_with_justify(JustifyText::Center),
        ))),
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

fn hotkeys(keys: Res<ButtonInput<KeyCode>>, colors: ResMut<Colors>, mut commands: Commands) {
    let mut flush = false;
    let mut guard = colors.0.write();
    if keys.just_pressed(KeyCode::Equal) {
        guard.push(random_color());
        flush = true;
    } else if keys.just_pressed(KeyCode::Minus) {
        guard.pop();
        flush = true;
    }
    if flush {
        commands.queue(colors.0.flush());
    }
}
