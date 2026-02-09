//! Key press counter with swappable save states, showcasing map reactivity.
mod utils;
use utils::*;

extern crate alloc;
use alloc::collections::BTreeMap;

use bevy_platform::collections::HashMap;

use bevy::{prelude::*, window::WindowResolution};
use jonmo::prelude::*;

const SAVE_CHARS: &str = "abcdefgh";

fn main() {
    let mut app = App::new();

    let save_states: HashMap<_, _> = SAVE_CHARS
        .chars()
        .map(|save_char| {
            (
                save_char,
                MutableBTreeMap::builder()
                    .values(
                        ROWS.iter()
                            .flat_map(|row| row.chars().map(|letter| (letter, LetterData::default())))
                            .collect::<BTreeMap<_, _>>(),
                    )
                    .spawn(app.world_mut()),
            )
        })
        .collect();

    app.add_plugins(examples_plugin)
        .insert_resource(SaveStates(save_states))
        .insert_resource(ActiveSave('a'))
        .add_systems(
            Startup,
            (
                |world: &mut World| {
                    ui_root().spawn(world);
                },
                camera,
            ),
        )
        .add_systems(Update, listen)
        .run();
}

const ROWS: [&str; 3] = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];

#[derive(Default, Clone, Debug)]
struct LetterData {
    count: usize,
    pressed: bool,
}

#[derive(Resource, Clone)]
struct SaveStates(HashMap<char, MutableBTreeMap<char, LetterData>>);

#[derive(Resource, Clone, Copy, PartialEq)]
struct ActiveSave(char);

fn get_active_map(save_states: &SaveStates, active_save: ActiveSave) -> MutableBTreeMap<char, LetterData> {
    save_states.0.get(&active_save.0).unwrap().clone()
}

const LETTER_SIZE: f32 = 60.;

fn ui_root() -> jonmo::Builder {
    let active_save = signal::from_resource_changed::<ActiveSave>();

    jonmo::Builder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        ..default()
    })
    .child(
        jonmo::Builder::from(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(GAP * 2.),
            padding: UiRect::all(Val::Px(GAP * 4.)),
            width: Val::Px(WindowResolution::default().physical_width() as f32),
            justify_content: JustifyContent::Center,
            ..default()
        })
        .child(
            jonmo::Builder::from(Node {
                align_self: AlignSelf::Center,
                width: Val::Percent(100.),
                ..default()
            })
            .child(
                jonmo::Builder::from(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(GAP * 2.),
                    ..default()
                })
                .children(SAVE_CHARS.chars().map(clone!((active_save) move |save_char| {
                    save_card(save_char, active_save.clone())
                }))),
            )
            .child(
                sum_container()
                .child(
                    text_node()
                        .insert((TextColor(BLUE), TextFont::from_font_size(LETTER_SIZE)))
                        .with_component::<Node>(|mut node| node.height = Val::Px(100.))
                        .component_signal(
                            active_save
                                .clone()
                                .switch_signal_vec(move |In(active_save): In<ActiveSave>, save_states: Res<SaveStates>| {
                                    get_active_map(&save_states, active_save).signal_vec_entries()
                                })
                                .map_in(|(_, LetterData { count, .. })| count)
                                .sum()
                                .dedupe()
                                .map_in_ref(ToString::to_string)
                                .map_in(Text)
                                .map_in(Some),
                        )
                ),
            ),
        )
        .children(ROWS.into_iter().map(clone!((active_save) move |row| {
            jonmo::Builder::from(Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(GAP * 2.),
                ..default()
            })
            .children(row.chars().map(
                clone!((active_save) move |l| {
                    let letter_data = active_save
                        .clone()
                        .switch_signal_map(move |In(active_save): In<ActiveSave>, save_states: Res<SaveStates>| {
                            get_active_map(&save_states, active_save).signal_map()
                        })
                        .key(l)
                        .map_in(Option::unwrap_or_default);
                    letter(l, letter_data)
                }),
            ))
            .child(
                sum_container()
                .child(
                    text_node()
                        .insert((TextColor(BLUE), TextFont::from_font_size(LETTER_SIZE)))
                        .component_signal(
                            active_save
                                .clone()
                                .switch_signal_vec(move |In(active_save): In<ActiveSave>, save_states: Res<SaveStates>| {
                                    get_active_map(&save_states, active_save).signal_vec_entries()
                                })
                                .filter(move |In((letter, _))| row.contains(letter))
                                .map_in(|(_, LetterData { count, .. })| count)
                                .sum()
                                .dedupe()
                                .map_in_ref(ToString::to_string)
                                .map_in(Text)
                                .map_in(Some),
                        ),
                ),
            )
        }))),
    )
}

const GAP: f32 = 5.;

fn sum_container() -> jonmo::Builder {
    jonmo::Builder::from(Node {
        align_self: AlignSelf::Center,
        justify_content: JustifyContent::FlexEnd,
        flex_grow: 1.,
        padding: UiRect::all(Val::Px(GAP * 2.)),
        ..default()
    })
}

fn save_card(save_char: char, active_save_signal: impl Signal<Item = ActiveSave> + Clone) -> jonmo::Builder {
    jonmo::Builder::from((
        Node {
            width: Val::Px(100.),
            height: Val::Px(100.),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            padding: UiRect::all(Val::Px(GAP * 2.)),
            ..default()
        },
        BorderRadius::all(Val::Px(GAP * 2.)),
    ))
    .observe(move |_click: On<Pointer<Click>>, mut active_save: ResMut<ActiveSave>| {
        active_save.0 = save_char;
    })
    .component_signal(
        active_save_signal
            .map_in(move |ActiveSave(active_char)| {
                if active_char == save_char {
                    BackgroundColor(BLUE)
                } else {
                    BackgroundColor(PINK)
                }
            })
            .map_in(Some),
    )
    .child(text_node().insert((
        Text(save_char.to_string()),
        TextColor(Color::WHITE),
        TextFont::from_font_size(LETTER_SIZE),
    )))
}

fn text_node() -> jonmo::Builder {
    jonmo::Builder::from((
        Node::default(),
        TextColor(Color::WHITE),
        TextLayout::new_with_justify(Justify::Center),
        BorderRadius::all(Val::Px(GAP)),
    ))
}

fn letter(letter: char, data: impl Signal<Item = LetterData> + Clone) -> jonmo::Builder {
    jonmo::Builder::from((
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(GAP * 2.),
            padding: UiRect::all(Val::Px(GAP * 2.)),
            width: Val::Px(100.),
            ..default()
        },
        BorderRadius::all(Val::Px(GAP * 2.)),
    ))
    .component_signal(
        data.clone()
            .map_in(|LetterData { pressed, .. }| pressed)
            .dedupe()
            .map_true_in(|| Outline {
                width: Val::Px(1.),
                ..default()
            }),
    )
    .child(text_node().insert((
        Text(letter.to_string()),
        TextColor(PINK),
        TextFont::from_font_size(LETTER_SIZE),
    )))
    .child(
        text_node()
            .insert(TextFont::from_font_size(LETTER_SIZE / 1.5))
            .component_signal(
                data.clone()
                    .map_in(|LetterData { count, .. }| count)
                    .dedupe()
                    .map_in_ref(ToString::to_string)
                    .map_in(Text)
                    .map_in(Some),
            ),
    )
}

fn listen(
    keys: ResMut<ButtonInput<KeyCode>>,
    save_states: Res<SaveStates>,
    active_save: Res<ActiveSave>,
    mut mutable_btree_map_datas: Query<&mut MutableBTreeMapData<char, LetterData>>,
) {
    let current_map = get_active_map(&save_states, *active_save);
    let map = HashMap::from([
        (KeyCode::KeyA, 'a'),
        (KeyCode::KeyB, 'b'),
        (KeyCode::KeyC, 'c'),
        (KeyCode::KeyD, 'd'),
        (KeyCode::KeyE, 'e'),
        (KeyCode::KeyF, 'f'),
        (KeyCode::KeyG, 'g'),
        (KeyCode::KeyH, 'h'),
        (KeyCode::KeyI, 'i'),
        (KeyCode::KeyJ, 'j'),
        (KeyCode::KeyK, 'k'),
        (KeyCode::KeyL, 'l'),
        (KeyCode::KeyM, 'm'),
        (KeyCode::KeyN, 'n'),
        (KeyCode::KeyO, 'o'),
        (KeyCode::KeyP, 'p'),
        (KeyCode::KeyQ, 'q'),
        (KeyCode::KeyR, 'r'),
        (KeyCode::KeyS, 's'),
        (KeyCode::KeyT, 't'),
        (KeyCode::KeyU, 'u'),
        (KeyCode::KeyV, 'v'),
        (KeyCode::KeyW, 'w'),
        (KeyCode::KeyX, 'x'),
        (KeyCode::KeyY, 'y'),
        (KeyCode::KeyZ, 'z'),
    ]);
    for (key, char) in map.iter() {
        if keys.just_pressed(*key) {
            let mut guard = current_map.write(&mut mutable_btree_map_datas);
            guard.insert(
                *char,
                LetterData {
                    pressed: true,
                    count: guard.get(char).unwrap().count + 1,
                },
            );
        } else if keys.just_released(*key) {
            let mut guard = current_map.write(&mut mutable_btree_map_datas);
            guard.insert(
                *char,
                LetterData {
                    pressed: false,
                    ..guard.get(char).unwrap().clone()
                },
            );
        }
    }
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
