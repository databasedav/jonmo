#![doc = include_str!("../README.md")]
//! ## feature flags
#![cfg_attr(
    feature = "document-features",
    doc = document_features::document_features!()
)]
#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;

use bevy_app::prelude::*;
use bevy_ecs::{
    prelude::*,
    schedule::{InternedScheduleLabel, ScheduleLabel},
};
use bevy_platform::prelude::*;

pub mod graph;
pub mod signal;
pub mod signal_map;
pub mod signal_vec;
pub mod utils;

cfg_if::cfg_if! {
    if #[cfg(feature = "builder")] {
        pub mod builder;
        #[doc(inline)]
        pub use builder::Builder;
    }
}

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
/// app.add_plugins(JonmoPlugin::new::<PostUpdate>());
///
/// // Add ordering constraints using configure_sets
/// #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
/// struct SystemSet1;
///
/// #[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
/// struct SystemSet2;
///
/// app.configure_sets(
///     PostUpdate,
///     SignalProcessing.before(SystemSet1).before(SystemSet2),
/// );
/// ```
///
/// # Multi-Schedule Processing
///
/// For advanced use cases where signals need to run in different schedules (e.g., to avoid
/// UI flicker by spawning in `Update` rather than `PostUpdate`), register multiple schedules:
///
/// ```rust
/// use bevy_app::prelude::*;
/// use jonmo::prelude::*;
///
/// # let mut app = App::new();
/// app.add_plugins(JonmoPlugin::new::<PostUpdate>().with_schedule::<Update>());
/// ```
///
/// Then use [`.schedule`](signal::SignalExt::schedule) on individual signal chains to control which
/// schedule they run in.
pub struct JonmoPlugin {
    schedules: Vec<InternedScheduleLabel>,
    registration_recursion_limit: usize,
    on_recursion_limit_exceeded: graph::RecursionLimitBehavior,
}

impl Default for JonmoPlugin {
    fn default() -> Self {
        Self::new::<Last>()
    }
}

impl JonmoPlugin {
    /// Create a new `JonmoPlugin` with the given default schedule.
    ///
    /// The default schedule is used for signals without explicit `.schedule::<S>()` calls.
    /// Additional schedules can be added with `.schedule::<S>()`.
    pub fn new<S: ScheduleLabel + Default>() -> Self {
        Self {
            schedules: vec![S::default().intern()],
            registration_recursion_limit: graph::DEFAULT_REGISTRATION_RECURSION_LIMIT,
            on_recursion_limit_exceeded: graph::RecursionLimitBehavior::default(),
        }
    }

    /// Add an additional schedule for signal processing.
    ///
    /// Use [`SignalExt::schedule`](crate::signal::SignalExt::schedule)
    /// on individual signal chains to control which schedule they run in.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bevy_app::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// # let mut app = App::new();
    /// app.add_plugins(
    ///     JonmoPlugin::new::<PostUpdate>() // Default schedule
    ///         .with_schedule::<Update>(), // Additional schedule
    /// );
    /// ```
    pub fn with_schedule<S: ScheduleLabel + Default>(mut self) -> Self {
        let interned = S::default().intern();
        if !self.schedules.contains(&interned) {
            self.schedules.push(interned);
        }
        self
    }

    /// Set the maximum number of signal registration recursion passes per frame.
    ///
    /// During signal processing, signals can spawn new signals (e.g., UI elements
    /// registering child signals). These new signals are processed in the same frame
    /// via recursive passes. This limit prevents infinite loops.
    ///
    /// Default: 100
    ///
    /// # Example
    ///
    /// ```rust
    /// use bevy_app::prelude::*;
    /// use jonmo::prelude::*;
    ///
    /// # let mut app = App::new();
    /// app.add_plugins(JonmoPlugin::new::<Last>().registration_recursion_limit(50));
    /// ```
    pub fn registration_recursion_limit(mut self, limit: usize) -> Self {
        self.registration_recursion_limit = limit;
        self
    }

    /// Set the behavior when the signal registration recursion limit is exceeded.
    ///
    /// Default: [`RecursionLimitBehavior::Panic`](graph::RecursionLimitBehavior::Panic)
    ///
    /// # Example
    ///
    /// ```rust
    /// use bevy_app::prelude::*;
    /// use jonmo::{graph::RecursionLimitBehavior, prelude::*};
    ///
    /// # let mut app = App::new();
    /// // In production, you might want to warn instead of panic
    /// app.add_plugins(
    ///     JonmoPlugin::new::<Last>().on_recursion_limit_exceeded(RecursionLimitBehavior::Warn),
    /// );
    /// ```
    pub fn on_recursion_limit_exceeded(mut self, behavior: graph::RecursionLimitBehavior) -> Self {
        self.on_recursion_limit_exceeded = behavior;
        self
    }
}

/// [`SystemSet`] that can be used to schedule systems around signal processing.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignalProcessing;

impl Plugin for JonmoPlugin {
    fn build(&self, app: &mut App) {
        // First schedule is the default for signals without explicit scheduling
        let default_schedule = self.schedules[0];

        // Initialize graph state with configuration
        app.insert_resource(graph::SignalGraphState::with_options(
            default_schedule,
            self.registration_recursion_limit,
            self.on_recursion_limit_exceeded,
        ));

        // Register processing system for each schedule
        for &schedule in &self.schedules {
            app.add_systems(
                schedule,
                (
                    (
                        signal_vec::trigger_replays::<signal_vec::VecReplayTrigger>,
                        signal_vec::trigger_replays::<signal_map::MapReplayTrigger>,
                    ),
                    graph::process_signal_graph_for_schedule(schedule),
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

        // Clear persistent inputs at frame end (after all signal processing)
        app.add_systems(Last, graph::clear_signal_inputs.after(SignalProcessing));
    }
}

/// `use jonmo::prelude::*;` imports everything one needs to use start using [jonmo](crate).
pub mod prelude {
    #[cfg(feature = "builder")]
    pub use crate::builder::{SignalMapTaskExt, SignalTask, SignalTaskExt, SignalVecTaskExt};
    pub use crate::{
        JonmoPlugin,
        graph::SignalHandles,
        signal::{self, BoxedSignal, IntoSignalEither, Signal, SignalEither, SignalExt},
        signal_map::{
            IntoSignalMapEither, MutableBTreeMap, MutableBTreeMapData, SignalMap, SignalMapEither, SignalMapExt,
        },
        signal_vec::{IntoSignalVecEither, MutableVec, MutableVecData, SignalVec, SignalVecEither, SignalVecExt},
        utils::{LazyEntity, clone, deref_cloned, deref_copied},
    };
    #[doc(no_inline)]
    pub use apply::{Also, Apply};
}
