use bevy_ecs::prelude::*;
use bevy_platform::sync::*;
#[doc(no_inline)]
pub use enclose::enclose as clone;

/// A deferred, thread-safe, clone-able handle to an [`Entity`]. Useful when the existence of an
/// entity is known at compile time but it can't be referenced until after it's spawned, e.g. in the
/// bodies of systems.
#[derive(Default, Clone)]
pub struct LazyEntity(Arc<OnceLock<Entity>>);

impl LazyEntity {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the [`Entity`], panicking if it was already set.
    pub fn set(&self, entity: Entity) {
        self.0.set(entity).expect("EntityHolder already contains an Entity");
    }

    /// Get the [`Entity`], panicking if it was not set.
    pub fn get(&self) -> Entity {
        self.0.get().copied().expect("EntityHolder does not contain an Entity")
    }
}

/// Convenience trait for [`Send`] + [`Sync`] + 'static.
pub trait SSs: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> SSs for T {}
