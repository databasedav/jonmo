# jonmo [জন্ম](https://translate.google.com/?sl=bn&tl=en&text=%E0%A6%9C%E0%A6%A8%E0%A7%8D%E0%A6%AE&op=translate)

[![Crates.io Version](https://img.shields.io/crates/v/jonmo?style=for-the-badge)](https://crates.io/crates/jonmo)
[![Docs.rs](https://img.shields.io/docsrs/jonmo?style=for-the-badge)](https://docs.rs/jonmo)
[![Following released Bevy versions](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue?style=for-the-badge)](https://bevyengine.org/learn/quick-start/plugin-development/#main-branch-tracking)

```text
in bengali, jonmo means "birth"
```

[jonmo](https://github.com/databasedav/jonmo) provides an ergonomic, functional, and declarative API for specifying Bevy [system](https://docs.rs/bevy/latest/bevy/ecs/system/index.html) dependency graphs, where "output" handles to nodes of the graph are canonically referred to as "signals". Building upon these signals, jonmo offers a high level [entity builder](https://docs.rs/jonmo/latest/jonmo/struct.JonmoBuilder.html) which enables one to declare reactive entities, components, and children using a familiar fluent syntax with semantics and API ported from the incredible [FRP](https://en.wikipedia.org/wiki/Functional_reactive_programming) signals of [futures-signals](https://github.com/Pauan/rust-signals) and its web UI dependents [MoonZoon](https://github.com/MoonZoon/MoonZoon) and [Dominator](https://github.com/Pauan/rust-dominator).

The runtime of jonmo is quite simple; every frame, the outputs of systems are forwarded to their dependants, recursively. The complexity and power of jonmo really emerges from its monadic signal combinators, defined within the [`SignalExt`](https://docs.rs/jonmo/latest/jonmo/trait.SignalExt.html), [`SignalVecExt`](https://docs.rs/jonmo/latest/jonmo/trait.SignalVecExt.html), and [`SignalMapExt`](https://docs.rs/jonmo/latest/jonmo/trait.SignalMapExt.html) traits (ported from futures-signals' traits of the same name), which internally manage special Bevy systems that allow for the declarative composition of complex data flows with minimalistic, high-level, signals-oriented methods.

### Assorted features:
- fine-grained reactivity for all entities, components, and children
- ***diff-less*** constant-time reactive updates for collections (available through `MutableVec` and `MutableBTreeMap`)
- automated system lifecycle management when using the builder API, simple component on-remove hook when not
- polling API for when one needs an escape hatch from the regular push-based output semantics (polling is used sparsely internally for some combinators)
- either wrappers (a la https://github.com/rayon-rs/either) and type-erased signals (via boxing) for cheap and flexible management of distinct signal types from different branches of logic
- `no_std` *always*

## examples
```rust no_run
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
        .add_systems(
            Update,
            (live.run_if(any_with_component::<Lifetime>), hotkeys),
        )
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
    .children_signal_vec(
        colors
            .enumerate()
            .map_in(|(index, color)| item(index, color)),
    )
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
```

### on the web

All examples are compiled to wasm for both webgl2 and webgpu (check [compatibility](<https://github.com/gpuweb/gpuweb/wiki/Implementation-Status#implementation-status>)) and deployed to github pages.

- [**`basic`**](https://github.com/databasedav/jonmo/blob/main/examples/basic.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/basic/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/basic/)

    a simple increasing timer, without using the entity builder, showcasing the quickest way to start using jonmo signals in existing Bevy apps

- [**`basic_builder`**](https://github.com/databasedav/jonmo/blob/main/examples/basic_builder.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/basic_builder/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/basic_builder/)

    a simple increasing timer, using the entity builder, showcasing the recommended idiomatic way to use jonmo signals

- [**`lifetime_list`**](https://github.com/databasedav/jonmo/blob/main/examples/lifetime_list.rs) [webgl2](https://databasedav.github.io/jonmo/examples/webgl2/lifetime_list/) [webgpu](https://databasedav.github.io/jonmo/examples/webgpu/lifetime_list/)

    the example above, a reactive enumerated list of colors, each with an independent lifetime timer

## Bevy compatibility

|bevy|jonmo|
|-|-|
|0.16|0.2|
|0.15|0.1|

## license
All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](https://github.com/databasedav/jonmo/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/databasedav/jonmo/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.

### your contributions
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
