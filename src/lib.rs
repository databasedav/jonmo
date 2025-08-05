#![doc = include_str!("../README.md")]
//! ## feature flags
#![cfg_attr(
    feature = "document-features",
    doc = document_features::document_features!()
)]
#![no_std]

extern crate alloc;

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

/// Includes the systems required for [jonmo](crate) to function.
#[derive(Default)]
pub struct JonmoPlugin;

impl Plugin for JonmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Last,
            (
                signal_vec::flush_mutable_vecs,
                (
                    signal_vec::trigger_replays::<signal_vec::VecReplayTrigger>,
                    signal_vec::trigger_replays::<signal_map::MapReplayTrigger>,
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
    #[cfg(feature = "builder")]
    pub use crate::builder::JonmoBuilder;
    pub use crate::{
        JonmoPlugin,
        graph::SignalHandles,
        signal::{IntoSignalEither, Signal, SignalBuilder, SignalEither, SignalExt},
        signal_map::{MutableBTreeMap, SignalMap, SignalMapExt},
        signal_vec::{IntoSignalVecEither, MutableVec, SignalVec, SignalVecEither, SignalVecExt},
        utils::{LazyEntity, clone},
    };
    #[doc(no_inline)]
    pub use apply::{Also, Apply};
}
