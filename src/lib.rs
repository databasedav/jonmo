#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

pub mod builder;
pub mod graph;
pub mod signal;
pub mod signal_map;
pub mod signal_vec;
pub mod utils;

use graph::*;

/// Includes the systems required for [jonmo](crate) to function.
#[derive(Default)]
pub struct JonmoPlugin;

impl Plugin for JonmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Last, (process_signal_graph, flush_cleanup_signals).chain());
    }
}

/// `use jonmo::prelude::*;` imports everything one needs to use start using [jonmo](crate).
pub mod prelude {
    pub use crate::{
        JonmoPlugin,
        builder::JonmoBuilder,
        graph::SignalHandles,
        signal::{IntoSignalEither, Signal, SignalBuilder, SignalEither, SignalExt},
        signal_map::{MutableBTreeMap, SignalMap, SignalMapExt},
        signal_vec::{MutableVec, SignalVec, SignalVecExt},
        utils::{LazyEntity, clone},
    };
}
