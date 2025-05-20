//! # jonmo - Declarative Signals for Bevy
//!
//! jonmo provides a way to define reactive signal chains in Bevy using a declarative
//! builder pattern. Signals originate from sources (like component changes, resource changes,
//! or specific entities) and can be transformed (`map`), combined (`combine_with`), or
//! deduplicated (`dedupe`).
//!
//! The core building block is the [`Signal`] trait, representing a value that changes over time.
//! Chains are constructed starting with methods like [`SignalBuilder::from_component`] or
//! [`SignalBuilder::from_resource`], followed by combinators like [`SignalExt::map`] or
//! [`SignalExt::combine_with`]. Signal chains must implement `Clone` to be used with combinators
//! like `combine_with` or to be cloned into closures.
//!
//! Finally, a signal chain is activated by calling [`SignalExt::register`], which registers
//! the necessary Bevy systems and returns a [`SignalHandle`] for potential cleanup.
//! Cleaning up a handle removes *all* systems created by that specific `register` call
//! by decrementing reference counts. If systems were shared with other signal chains, cleaning up
//! one handle will only remove those shared systems if their reference count reaches zero.
//!
//! ## Execution Model
//!
//! Internally, jonmo builds and maintains a dependency graph of Bevy systems. Each frame,
//! the [`JonmoPlugin`] triggers the execution of this graph starting from the root systems
//! (created via `SignalBuilder::from_*` methods). It pipes the output (`Some(O)`) of a parent
//! system as the input (`In<O>`) to its children using type-erased runners. This traversal
//! continues down each branch until a system returns `None` (often represented by the
//! [`TERMINATE`] constant), which halts propagation along that specific path for the current frame.
//!
//! The signal propagation is managed internally by the [`JonmoPlugin`] which should be added
//! to your Bevy `App`.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

pub mod builder;
pub mod signal;
pub mod signal_vec;
pub mod tree;
pub mod utils;
pub use builder::JonmoBuilder;

use tree::*;

/// The Bevy plugin required for `jonmo` signals to function.
///
/// Adds the necessary [`SignalPropagator`] resource and the system that drives
/// signal propagation ([`process_signals`]) to the `Update` schedule.
///
/// ```no_run
/// use bevy::prelude::*;
/// use jonmo::prelude::*; // Use prelude
///
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(JonmoPlugin) // Add the plugin here
///     // ... other app setup ...
///     .run();
/// ```
#[derive(Default)]
pub struct JonmoPlugin;

impl Plugin for JonmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Last, (process_signals, flush_cleanup_signals).chain());
    }
}

/// Commonly used items for working with `jonmo` signals.
///
/// This prelude includes the core traits, structs, and functions needed to
/// define and manage signal chains. It excludes internal implementation details
/// like [`SignalBuilderInternal`].
///
/// ```
/// use jonmo::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        JonmoPlugin,
        builder::{JonmoBuilder, SignalHandles},
        signal::{IntoSignalEither, Signal, SignalBuilder, SignalEither, SignalExt},
        signal_vec::{MutableVec, SignalVec, SignalVecExt},
    };
}
