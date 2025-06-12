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
            (live.run_if(any_with_component::<Lifetime>), hotkeys, toggle),
        )
        .insert_resource(ToggleFilter(true))
        .run();
}

#[derive(Resource, Clone)]
struct Numbers(MutableVec<u32>);

#[derive(Component, Default, Clone, Reflect)]
struct Lifetime(f32);

fn ui_root(numbers: impl SignalVec<Item = u32> + Clone) -> JonmoBuilder {
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
        numbers
            .filter_signal(|In(n)| {
                SignalBuilder::from_system(move |_: In<()>, toggle: Res<ToggleFilter>| {
                    println!("toggle {}: {}", n, toggle.0);
                    if toggle.0 { n % 2 == 0 } else { true }
                })
            })
            .map(|In(color)| item(color)),
    )
}

#[derive(Resource)]
struct ToggleFilter(bool);

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
        guard.push((guard.len() + 1) as u32);
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
        info!("Toggle filter: {}", toggle.0);
    }
}

use bevy::{
    ecs::system::SystemId,
    prelude::{
        App, Commands, Component, DefaultPlugins, Entity, In, Local, Query, Startup, SystemInput,
        Update, With, World, info,
    },
};
use std::{marker::PhantomData, sync::Arc};

pub struct Plugin;
impl bevy::prelude::Plugin for Plugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (startup,));
        app.add_systems(Update, (run_pipes));
    }
}

#[derive(Clone)]
struct MyStruct {
    field: i32,
}

fn sys_1(commands: Commands) -> MyStruct {
    println!("sys1");
    MyStruct { field: 1 }
}

fn sys_2(In(my_struct): In<MyStruct>) {
    println!("field2: {}", my_struct.field);
}

fn sys_3(In(my_struct): In<MyStruct>) -> usize {
    println!("field3: {}", my_struct.field);
    1
}

#[derive(Component, Clone)]
struct Roots(Arc<dyn RunnableBadName<()> + Send + Sync + 'static>);

trait RunnableBadName<I: SystemInput + 'static> {
    fn run(&self, input: In<I::Inner<'static>>, world: &mut World);
}

#[derive(Component)]
struct Piped<I: SystemInput + 'static, O: 'static> {
    input: PhantomData<I>,
    input_sys: SystemId<I, O>,
    outs: Vec<Box<dyn RunnableBadName<In<O>> + Send + Sync + 'static>>,
}

impl<T: SystemInput + 'static, O: 'static + Clone> RunnableBadName<T> for Piped<T, O> {
    fn run(&self, input: In<T::Inner<'static>>, world: &mut World) {
        let res = world.run_system_with(self.input_sys, input.0).unwrap();
        for out in self.outs.iter() {
            out.run(In(res.clone()), world)
        }
    }
}
fn startup(mut commands: Commands) {
    commands.queue(|world: &mut World| {
        let id1: SystemId<(), MyStruct> = world.register_system(sys_1);

        let id2: SystemId<In<MyStruct>, ()> = world.register_system(sys_2);
        let id3: SystemId<In<MyStruct>, ()> = world.register_system(sys_3);

        let pipe_str = Piped {
            input: PhantomData,
            input_sys: id1,
            outs: vec![
                Box::new(Piped {
                    input: PhantomData,
                    input_sys: id2,
                    outs: vec![],
                }),
                Box::new(Piped {
                    input: PhantomData,
                    input_sys: id3,
                    outs: vec![],
                }),
            ],
        };
        world.spawn(Roots(Arc::new(pipe_str)));
    });
}

fn run_pipes(mut commands: Commands, mut local: Local<u32>, pipes: Query<Entity, With<Roots>>) {
    if *local < 10 {
        *local += 1;

        info!("Running iteration {}", *local);
        for pipe in pipes.iter() {
            commands.queue(move |world: &mut World| {
                let c = world.entity(pipe).get::<Roots>().cloned();
                c.unwrap().0.run(In(()), world);
            });
        }
    }
}
