#![doc = include_str!("../README.md")]
//! ## feature flags
#![cfg_attr(
    feature = "document-features",
    doc = document_features::document_features!()
)]
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

#[cfg(feature = "builder")]
pub mod builder;
pub mod graph;
pub mod signal;
pub mod signal_map;
pub mod signal_vec;
#[allow(missing_docs)]
pub mod utils;

fn trigger_replays<ReplayTrigger: Component + signal_vec::Replayable>(world: &mut World) {
    let triggers: Vec<Entity> = world
        .query_filtered::<Entity, With<ReplayTrigger>>()
        .iter(world)
        .collect();
    for trigger_entity in triggers {
        if let Some(trigger_component) = world
            .get_entity_mut(trigger_entity)
            .ok()
            .and_then(|mut e| e.take::<ReplayTrigger>())
        {
            trigger_component.trigger()(world);
        }
    }
}

/// Includes the systems required for [jonmo](crate) to function.
#[derive(Default)]
pub struct JonmoPlugin;

impl Plugin for JonmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Last,
            (
                (
                    trigger_replays::<signal_vec::VecReplayTrigger>,
                    trigger_replays::<signal_map::MapReplayTrigger>,
                ),
                graph::process_signal_graph,
                graph::flush_cleanup_signals,
            )
                .chain(),
        );
    }
}

/// `use jonmo::prelude::*;` imports everything one needs to use start using [jonmo](crate).
pub mod prelude {
    pub use crate::{
        JonmoPlugin,
        graph::SignalHandles,
        signal::{IntoSignalEither, Signal, SignalBuilder, SignalEither, SignalExt},
        signal_map::{MutableBTreeMap, SignalMap, SignalMapExt},
        signal_vec::{IntoSignalVecEither, MutableVec, SignalVec, SignalVecEither, SignalVecExt},
        utils::{LazyEntity, clone},
    };
    #[cfg(feature = "builder")]
    pub use crate::builder::JonmoBuilder;
    #[doc(no_inline)]
    pub use apply::{Also, Apply};
}
