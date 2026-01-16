#![doc = include_str!("../README.md")]
//! ## feature flags
#![cfg_attr(
    feature = "document-features",
    doc = document_features::document_features!()
)]
#![cfg_attr(not(test), no_std)]

extern crate alloc;

use bevy_app::prelude::*;
use bevy_ecs::{
    prelude::*,
    schedule::{InternedScheduleLabel, ScheduleLabel},
};

#[cfg(feature = "builder")]
pub mod builder;
pub mod graph;
pub mod signal;
pub mod signal_map;
pub mod signal_vec;
#[allow(missing_docs)]
pub mod utils;

/// Includes the systems required for [jonmo](crate) to function.
///
/// # Example
///
/// ```rust
/// use bevy_app::prelude::*;
/// use bevy_ecs::prelude::*;
/// use jonmo::{SignalProcessing, prelude::*};
///
/// let mut app = App::new();
/// // Use default configuration (runs in Last schedule)
/// app.add_plugins(JonmoPlugin::default());
///
/// let mut app = App::new();
/// // Or customize the schedule
/// app.add_plugins(JonmoPlugin::new().in_schedule(PostUpdate));
///
/// // Add ordering constraints using configure_sets
/// # #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
/// # struct SystemSet1;
///
/// # #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
/// # struct SystemSet2;
///
/// app.configure_sets(
///     PostUpdate,
///     SignalProcessing.before(SystemSet1).before(SystemSet2),
/// );
/// ```
pub struct JonmoPlugin {
    schedule: InternedScheduleLabel,
}

impl Default for JonmoPlugin {
    fn default() -> Self {
        Self {
            schedule: Last.intern(),
        }
    }
}

impl JonmoPlugin {
    /// Create a new `JonmoPlugin` with signal processing running in the `Last` schedule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Specify which schedule the signal processing systems should run in.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bevy_app::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// JonmoPlugin::new().in_schedule(Update);
    /// ```
    pub fn in_schedule(mut self, schedule: impl ScheduleLabel) -> Self {
        self.schedule = schedule.intern();
        self
    }
}

/// [`SystemSet`] that can be used to schedule systems around signal processing.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignalProcessing;

impl Plugin for JonmoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<graph::SignalGraphState>();
        app.add_systems(
            self.schedule,
            (
                (
                    signal_vec::trigger_replays::<signal_vec::VecReplayTrigger>,
                    signal_vec::trigger_replays::<signal_map::MapReplayTrigger>,
                ),
                graph::process_signal_graph,
                (
                    graph::despawn_stale_signals,
                    signal_vec::despawn_stale_mutable_vecs,
                    signal_map::despawn_stale_mutable_btree_maps,
                ),
            )
                .chain()
                .in_set(SignalProcessing),
        );
    }
}

/// `use jonmo::prelude::*;` imports everything one needs to use start using [jonmo](crate).
pub mod prelude {
    #[cfg(feature = "builder")]
    pub use crate::builder::{Holdable, JonmoBuilder, SignalHoldExt, SignalMapHoldExt, SignalVecHoldExt};
    pub use crate::{
        JonmoPlugin,
        graph::SignalHandles,
        signal::{self, IntoSignalEither, Signal, SignalBuilder, SignalEither, SignalExt},
        signal_map::{
            IntoSignalMapEither, MutableBTreeMap, MutableBTreeMapBuilder, MutableBTreeMapData, SignalMap,
            SignalMapEither, SignalMapExt,
        },
        signal_vec::{
            IntoSignalVecEither, MutableVec, MutableVecBuilder, MutableVecData, SignalVec, SignalVecEither,
            SignalVecExt,
        },
        utils::{LazyEntity, clone, deref_cloned, deref_copied},
    };
    #[doc(no_inline)]
    pub use apply::{Also, Apply};
}
