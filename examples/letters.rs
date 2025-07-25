//! Diverse filtering options for a list of items, showcasing the power of vector signals.
mod utils;
use utils::*;

extern crate alloc;
use alloc::collections::BTreeMap;

use bevy_platform::collections::HashMap;

use bevy::prelude::*;
use jonmo::prelude::*;

fn main() {
    let mut app = App::new();
    let letters = MutableBTreeMap::from(
        ROWS.iter()
            .flat_map(|row| row.chars().map(|letter| (letter, LetterData::default())))
            .collect::<BTreeMap<_, _>>(),
    );
    app.add_plugins(examples_plugin)
        .insert_resource(Letters(letters.clone()))
        .add_systems(
            Startup,
            (
                move |world: &mut World| {
                    ui_root(letters.clone()).spawn(world);
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
struct Letters(MutableBTreeMap<char, LetterData>);

const LETTER_SIZE: f32 = 60.;

fn ui_root(letters: MutableBTreeMap<char, LetterData>) -> JonmoBuilder {
    JonmoBuilder::from(Node {
        height: Val::Percent(100.0),
        width: Val::Percent(100.0),
        ..default()
    })
    .child(
        JonmoBuilder::from(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(GAP * 2.),
            padding: UiRect::all(Val::Px(GAP * 2.)),
            width: Val::Percent(100.),
            justify_content: JustifyContent::Center,
            ..default()
        })
        .child(
            JonmoBuilder::from(Node {
                align_self: AlignSelf::Center,
                justify_content: JustifyContent::FlexEnd,
                width: Val::Percent(100.),
                // height: Val::Px(100.),
                padding: UiRect::all(Val::Px(GAP * 2.)),
                ..default()
            })
            .child(
                text_node()
                    .insert(TextFont::from_font_size(LETTER_SIZE))
                    .with_component::<Node>(|mut node| node.height = Val::Px(100.))
                    .component_signal(
                        letters
                            .signal_vec_entries()
                            .map_in(|(_, LetterData { count, .. })| count)
                            .sum()
                            .dedupe()
                            .map_in_ref(ToString::to_string)
                            .map_in(Text)
                            .map_in(Some),
                    ),
            ),
        )
        .children(ROWS.into_iter().map(move |row| {
            JonmoBuilder::from(Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(GAP * 2.),
                ..default()
            })
            .children(row.chars().map(
                clone!((letters) move |l| letter(l, letters.signal_map().debug().key(l).map_in(Option::unwrap_or_default))),
            ))
            .child(
                JonmoBuilder::from(Node {
                    align_self: AlignSelf::Center,
                    justify_content: JustifyContent::FlexEnd,
                    flex_grow: 1.,
                    padding: UiRect::all(Val::Px(GAP * 2.)),
                    ..default()
                })
                .child(
                    text_node()
                        .insert(TextFont::from_font_size(LETTER_SIZE))
                        .component_signal(
                            letters
                                .signal_vec_entries()
                                .filter(|In((letter, _))| row.contains(letter))
                                .map_in(|(_, LetterData { count, .. })| count)
                                .sum()
                                .dedupe()
                                .map_in_ref(ToString::to_string)
                                .map_in(Text)
                                .map_in(Some),
                        ),
                ),
            )
        })),
    )
}

const GAP: f32 = 5.;

fn text_node() -> JonmoBuilder {
    JonmoBuilder::from((
        Node::default(),
        TextColor(Color::WHITE),
        TextLayout::new_with_justify(JustifyText::Center),
        BorderRadius::all(Val::Px(GAP)),
    ))
}

fn letter(letter: char, data: impl Signal<Item = LetterData> + Clone) -> JonmoBuilder {
    JonmoBuilder::from((
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(GAP * 2.),
            padding: UiRect::all(Val::Px(GAP * 2.)),
            width: Val::Px(100.),
            ..default()
        },
        BorderRadius::all(Val::Px(GAP * 2.)),
    ))
    .with_entity(|mut entity| {
        entity.observe(|click: Trigger<Pointer<Click>>, nodes: Query<&ComputedNode>| {
            println!("{}", nodes.get(click.target()).unwrap().size());
        });
    })
    .component_signal(
        data.clone()
            .map_in(|LetterData { pressed, .. }| pressed)
            .dedupe()
            .map_true(|_: In<()>| Outline {
                width: Val::Px(1.),
                ..default()
            }),
    )
    .child(text_node().insert((Text(letter.to_string()), TextFont::from_font_size(LETTER_SIZE))))
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

fn listen(keys: ResMut<ButtonInput<KeyCode>>, letters: Res<Letters>, mut commands: Commands) {
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
    let mut flush = false;
    for (key, char) in map.iter() {
        if keys.just_pressed(*key) {
            let mut guard = letters.0.write();
            guard.insert(
                *char,
                LetterData {
                    pressed: true,
                    count: guard.get(char).unwrap().count + 1,
                },
            );
            flush = true;
        } else if keys.just_released(*key) {
            let mut guard = letters.0.write();
            guard.insert(
                *char,
                LetterData {
                    pressed: false,
                    ..guard.get(char).unwrap().clone()
                },
            );
            flush = true;
        }
    }
    if flush {
        commands.queue(letters.0.flush());
    }
}

fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
