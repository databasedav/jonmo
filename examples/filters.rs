//! Diverse filtering options for a list of items, showcasing vector reactivity.

mod utils;
use utils::*;

use bevy::{ecs::system::IntoObserverSystem, platform::collections::HashSet, prelude::*};
use jonmo::{prelude::*, utils::SSs};
use rand::{Rng, prelude::IndexedRandom};

fn main() {
    let mut app = App::new();
    let world = app.world_mut();
    let datas = MutableVecBuilder::from((0..12).map(|_| random_data()).collect::<Vec<_>>()).spawn(world);
    let rows = MutableVecBuilder::from((0..5).map(|_| ()).collect::<Vec<_>>()).spawn(world);
    app.add_plugins(examples_plugin)
        .insert_resource(Datas(datas.clone()))
        .insert_resource(Rows(rows.clone()))
        .add_systems(
            Startup,
            (
                move |world: &mut World| {
                    ui(datas.clone(), rows.clone()).spawn(world);
                },
                camera,
            ),
        )
        .run();
}

fn random_data() -> Data {
    let mut rng = rand::rng();
    Data {
        number: rng.random_range(..100),
        color: [ColorEnum::Blue, ColorEnum::Pink, ColorEnum::White]
            .choose(&mut rng)
            .copied()
            .unwrap(),
        shape: [Shape::Square, Shape::Circle].choose(&mut rng).copied().unwrap(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum ColorEnum {
    Blue,
    Pink,
    White,
}

impl From<ColorEnum> for Color {
    fn from(val: ColorEnum) -> Self {
        match val {
            ColorEnum::Blue => BLUE,
            ColorEnum::Pink => PINK,
            ColorEnum::White => Color::WHITE,
        }
    }
}

#[derive(Clone, Debug)]
struct Data {
    number: u32,
    color: ColorEnum,
    shape: Shape,
}

#[derive(Resource)]
struct Datas(MutableVec<Data>);

#[derive(Resource)]
struct Rows(MutableVec<()>);

#[derive(Component, Clone, PartialEq, Debug)]
struct NumberFilters(HashSet<Parity>);

#[derive(Component, Clone, PartialEq)]
struct ColorFilters(HashSet<ColorEnum>);

#[derive(Component, Clone, PartialEq)]
struct ShapeFilters(HashSet<Shape>);

#[derive(Component, Clone)]
struct Sorted;

const GAP: f32 = 5.;

fn ui(items: MutableVec<Data>, rows: MutableVec<()>) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        height: Val::Percent(100.),
        width: Val::Percent(100.),
        ..default()
    })
    .child(
        JonmoBuilder::from(Node {
            flex_direction: FlexDirection::Column,
            align_self: AlignSelf::Start,
            justify_self: JustifySelf::Start,
            row_gap: Val::Px(GAP * 2.),
            padding: UiRect::all(Val::Px(GAP * 4.)),
            ..default()
        })
        .child(
            JonmoBuilder::from((Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(GAP * 2.),
                ..default()
            },))
            .child(JonmoBuilder::from((
                Node::default(),
                Text::new("source"),
                TextColor(Color::WHITE),
                TextFont::from_font_size(30.),
                TextLayout::new_with_justify(Justify::Center),
            )))
            .child(button("+", -2.).apply(on_click(
                |_: On<Pointer<Click>>,
                 datas: Res<Datas>,
                 mut mutable_vec_datas: Query<&mut MutableVecData<_>>| {
                    datas.0.write(&mut mutable_vec_datas).insert(0, random_data());
                },
            )))
            .child(
                JonmoBuilder::from(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(GAP),
                    ..default()
                })
                .children_signal_vec(
                    items
                        .signal_vec()
                        .enumerate()
                        .map_in(|(index, data)| item(index.dedupe(), data)),
                ),
            ),
        )
        .children_signal_vec(
            rows.signal_vec()
                .enumerate()
                .map_in(clone!((items) move |(index, _)| row(index.dedupe(), items.clone()))),
        )
        .child(
            JonmoBuilder::from((
                Node {
                    height: Val::Px(55.),
                    justify_content: JustifyContent::Center,
                    flex_direction: FlexDirection::Column,
                    ..default()
                },
                // BackgroundColor(Color::WHITE),
            ))
            .child(button("+", -2.).apply(on_click(
                |_: On<Pointer<Click>>, rows: Res<Rows>, mut mutable_vec_datas: Query<&mut MutableVecData<_>>| {
                    rows.0.write(&mut mutable_vec_datas).push(());
                },
            ))),
        ),
    )
}

fn random_subset<T: Clone>(items: &[T]) -> Vec<T> {
    let mut rng = rand::rng();
    loop {
        let subset: Vec<T> = items
            .iter()
            .filter(|_| rng.random_bool(0.5)) // "Flip a coin" for each item
            .cloned() // Convert from `&&T` to `T`
            .collect();

        if !subset.is_empty() {
            // If we have at least one item, return the subset
            return subset;
        }
        // Otherwise, the loop continues and we try again
    }
}

fn random_number_filters() -> NumberFilters {
    NumberFilters(HashSet::from_iter(random_subset(&[Parity::Odd, Parity::Even])))
}

fn random_color_filters() -> ColorFilters {
    ColorFilters(HashSet::from_iter(random_subset(&[
        ColorEnum::Blue,
        ColorEnum::Pink,
        ColorEnum::White,
    ])))
}

fn random_shape_filters() -> ShapeFilters {
    ShapeFilters(HashSet::from_iter(random_subset(&[Shape::Square, Shape::Circle])))
}

fn maybe_insert_random_sorted(builder: JonmoBuilder) -> JonmoBuilder {
    let mut rng = rand::rng();
    if rng.random_bool(0.5) {
        builder.insert(Sorted)
    } else {
        builder
    }
}

fn text_node(text: &'static str) -> JonmoBuilder {
    JonmoBuilder::from((
        Node::default(),
        Text::new(text),
        TextColor(Color::WHITE),
        TextLayout::new_with_justify(Justify::Center),
        BorderRadius::all(Val::Px(GAP)),
    ))
}

fn toggle<T: Eq + core::hash::Hash>(set: &mut HashSet<T>, value: T) {
    if !set.remove(&value) {
        set.insert(value);
    }
}

fn on_click<M>(
    on_click: impl IntoObserverSystem<Pointer<Click>, (), M> + SSs,
) -> impl FnOnce(JonmoBuilder) -> JonmoBuilder {
    move |builder: JonmoBuilder| {
        builder.on_spawn(move |world, entity| {
            world.entity_mut(entity).observe(on_click);
        })
    }
}

fn outline() -> Outline {
    Outline {
        width: Val::Px(1.),
        ..default()
    }
}

fn number_toggle(row_parent: LazyEntity, parity: Parity) -> impl Fn(JonmoBuilder) -> JonmoBuilder {
    move |builder| {
        builder
            .apply(on_click(
                clone!((row_parent) move |_: On<Pointer<Click>>, mut number_filters: Query<&mut NumberFilters>| {
                    toggle(&mut number_filters.get_mut(row_parent.get()).unwrap().0, parity);
                }),
            ))
            .component_signal(
                SignalBuilder::from_component_lazy(row_parent.clone())
                    .dedupe()
                    .map_in(move |NumberFilters(filters)| filters.contains(&parity))
                    .dedupe()
                    .map_true(|_: In<()>| outline()),
            )
    }
}

fn number_toggles(row_parent: LazyEntity) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(2.),
        ..default()
    })
    .child(
        text_node("even")
            .insert(TextFont::from_font_size(13.))
            .insert(BackgroundColor(bevy::color::palettes::basic::GRAY.into()))
            .apply(number_toggle(row_parent.clone(), Parity::Even)),
    )
    .child(
        text_node("odd")
            .insert(TextFont::from_font_size(13.))
            .insert(BackgroundColor(bevy::color::palettes::basic::GRAY.into()))
            .apply(number_toggle(row_parent.clone(), Parity::Odd)),
    )
    .child(
        text_node("sort")
            .insert(TextFont::from_font_size(13.))
            .insert(BackgroundColor(bevy::color::palettes::basic::GRAY.into()))
            .apply(on_click(
                clone!((row_parent) move |_: On<Pointer<Click>>, world: &mut World| {
                    let mut entity = world.entity_mut(row_parent.get());
                    if entity.take::<Sorted>().is_none() { entity.insert(Sorted); }
                }),
            ))
            .component_signal(
                SignalBuilder::from_lazy_entity(row_parent.clone())
                    .has_component::<Sorted>()
                    .dedupe()
                    .map_true(|_: In<()>| outline()),
            ),
    )
}

fn shape_toggle(row_parent: LazyEntity, shape: Shape) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            width: Val::Px(20.),
            height: Val::Px(20.),
            ..default()
        },
        BackgroundColor(bevy::color::palettes::basic::GRAY.into()),
    ))
    .apply(on_click(
        clone!((row_parent) move |_: On<Pointer<Click>>, mut shape_filters: Query<&mut ShapeFilters>| {
            toggle(&mut shape_filters.get_mut(row_parent.get()).unwrap().0, shape);
        }),
    ))
    .component_signal(
        SignalBuilder::from_component_lazy(row_parent.clone())
            .dedupe()
            .map_in(move |ShapeFilters(filters)| filters.contains(&shape))
            .dedupe()
            .map_true(|_: In<()>| outline()),
    )
}

fn shape_toggles(row_parent: LazyEntity) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        flex_direction: FlexDirection::Column,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(GAP),
        ..default()
    })
    .child(shape_toggle(row_parent.clone(), Shape::Square))
    .child(shape_toggle(row_parent.clone(), Shape::Circle).insert(BorderRadius::MAX))
}

fn color_toggles(row_parent: LazyEntity) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        flex_direction: FlexDirection::Column,
        justify_content: JustifyContent::Center,
        row_gap: Val::Px(GAP),
        ..default()
    })
    .children(
        [ColorEnum::Blue, ColorEnum::Pink, ColorEnum::White]
            .into_iter()
            .map(move |color| {
                JonmoBuilder::from((
                    Node {
                        width: Val::Px(15.),
                        height: Val::Px(15.),
                        border: UiRect::all(Val::Px(1.)),
                        ..default()
                    },
                    BorderRadius::all(Val::Px(GAP)),
                    BackgroundColor(color.into()),
                    BorderColor::all(Color::BLACK),
                ))
                .apply(on_click(
                    clone!((row_parent) move |_: On<Pointer<Click>>, mut color_filters: Query<&mut ColorFilters>| {
                        toggle(&mut color_filters.get_mut(row_parent.get()).unwrap().0, color);
                    }),
                ))
                .component_signal(
                    SignalBuilder::from_component_lazy(row_parent.clone())
                        .dedupe()
                        .map_in(move |ColorFilters(filters)| filters.contains(&color))
                        .dedupe()
                        .map_true(|_: In<()>| outline()),
                )
            }),
    )
}

fn button(text: &'static str, offset: f32) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            width: Val::Px((ITEM_SIZE / 2) as f32),
            height: Val::Px((ITEM_SIZE / 2) as f32),
            justify_content: JustifyContent::Center,
            border: UiRect::all(Val::Px(1.)),
            ..default()
        },
        BackgroundColor(bevy::color::palettes::basic::GRAY.into()),
        BorderColor::all(Color::WHITE),
        BorderRadius::all(Val::Px(GAP)),
    ))
    .child(
        text_node(text)
            .with_component::<Node>(move |mut node| node.top = Val::Px(offset))
            .insert(TextFont::from_font_size(24.)),
    )
}

#[derive(Component, Clone)]
struct Index(usize);

fn row(index: impl Signal<Item = Option<usize>>, items: MutableVec<Data>) -> JonmoBuilder {
    let row_parent = LazyEntity::new();
    JonmoBuilder::from((
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(GAP * 2.),
            ..default()
        },
        random_number_filters(),
        random_color_filters(),
        random_shape_filters(),
    ))
    .apply(maybe_insert_random_sorted)
    .entity_sync(row_parent.clone())
    .child(
        button("-", -3.)
            .component_signal(index.map_in(|index| index.map(Index)))
            .apply(on_click(
                |click: On<Pointer<Click>>,
                 rows: Res<Rows>,
                 indices: Query<&Index>,
                 mut mutable_vec_datas: Query<&mut MutableVecData<_>>| {
                    if let Ok(&Index(index)) = indices.get(click.event().event_target()) {
                        rows.0.write(&mut mutable_vec_datas).remove(index);
                    }
                },
            )),
    )
    .child(
        JonmoBuilder::from((Node {
            flex_direction: FlexDirection::Row,
            width: Val::Px(108.),
            height: Val::Percent(100.),
            column_gap: Val::Px(GAP * 2.),
            justify_content: JustifyContent::Center,
            ..default()
        },))
        .child(number_toggles(row_parent.clone()))
        .child(shape_toggles(row_parent.clone()))
        .child(color_toggles(row_parent.clone())),
    )
    .child(
        JonmoBuilder::from((Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(GAP),
            ..default()
        },))
        .children_signal_vec(
            SignalBuilder::from_lazy_entity(row_parent.clone())
                .has_component::<Sorted>()
                .dedupe()
                .switch_signal_vec(move |In(sorted)| {
                    let base = items.signal_vec().enumerate();
                    if sorted {
                        base.sort_by_key(|In((_, Data { number, .. }))| number).left_either()
                    } else {
                        base.right_either()
                    }
                })
                .filter_signal(clone!((row_parent) move | In((_, Data { number, .. })) | {
                    SignalBuilder::from_component_lazy(row_parent.clone())
                        .dedupe()
                        .map_in(move |number_filters: NumberFilters| {
                            number_filters.0.contains(&if number.is_multiple_of(2) {
                                Parity::Even
                            } else {
                                Parity::Odd
                            })
                        })
                        .dedupe()
                }))
                .filter_signal(clone!((row_parent) move | In((_, Data { shape, .. })) | {
                    SignalBuilder::from_component_lazy(row_parent.clone())
                        .dedupe()
                        .map_in(move |shape_filters: ShapeFilters| shape_filters.0.contains(&shape))
                        .dedupe()
                }))
                .filter_signal(clone!((row_parent) move | In((_, Data { color, .. })) | {
                    SignalBuilder::from_component_lazy(row_parent.clone())
                        .dedupe()
                        .map_in(move |color_filters: ColorFilters| color_filters.0.contains(&color))
                        .dedupe()
                }))
                .map_in(|(index, data)| item(index.dedupe(), data)),
        ),
    )
}

const ITEM_SIZE: u32 = 50;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Shape {
    Square,
    Circle,
}

#[derive(Clone, Copy, PartialEq, Hash, Eq, Debug)]
enum Parity {
    Even,
    Odd,
}

fn item(index: impl Signal<Item = Option<usize>>, Data { number, color, shape }: Data) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            height: Val::Px(ITEM_SIZE as f32),
            width: Val::Px(ITEM_SIZE as f32),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        BackgroundColor(color.into()),
        match shape {
            Shape::Square => BorderRadius::default(),
            Shape::Circle => BorderRadius::MAX,
        },
    ))
    .component_signal(index.map_in(|index| index.map(Index)))
    .apply(on_click(
        |click: On<Pointer<Click>>,
         datas: Res<Datas>,
         indices: Query<&Index>,
         mut mutable_vec_datas: Query<&mut MutableVecData<_>>| {
            if let Ok(&Index(index)) = indices.get(click.event().event_target()) {
                datas.0.write(&mut mutable_vec_datas).remove(index);
            }
        },
    ))
    .child((
        Node::default(),
        Text::new(number.to_string()),
        TextColor(Color::BLACK),
        TextLayout::new_with_justify(Justify::Center),
    ))
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
