use core::ops::Deref;

use bevy_ecs::prelude::*;
use bevy_platform::sync::{Arc, OnceLock};
#[doc(no_inline)]
pub use enclose::enclose as clone;

/// A deferred, thread-safe, clone-able handle to an [`Entity`]. Useful when the existence of an
/// entity is known at compile time but it can't be referenced until after it's spawned, e.g. in the
/// bodies of systems.
#[derive(Default, Clone)]
pub struct LazyEntity(Arc<OnceLock<Entity>>);

const LAZY_ENTITY_GET_ERROR: &str = "LazyEntity does not contain an Entity";

impl LazyEntity {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the [`Entity`], panicking if it was already set.
    #[track_caller]
    pub fn set(&self, entity: Entity) {
        self.0.set(entity).expect("LazyEntity already contains an Entity");
    }

    /// Get the [`Entity`], panicking if it was not set.
    #[track_caller]
    pub fn get(&self) -> Entity {
        self.0.get().copied().expect(LAZY_ENTITY_GET_ERROR)
    }
}

impl Deref for LazyEntity {
    type Target = Entity;

    #[track_caller]
    fn deref(&self) -> &Self::Target {
        self.0.get().expect(LAZY_ENTITY_GET_ERROR)
    }
}

impl From<LazyEntity> for Entity {
    #[track_caller]
    fn from(lazy: LazyEntity) -> Entity {
        lazy.get()
    }
}

pub(crate) fn get_ancestor(child_ofs: &Query<&ChildOf>, entity: Entity, generations: usize) -> Option<Entity> {
    [entity]
        .into_iter()
        .chain(child_ofs.iter_ancestors(entity))
        .nth(generations)
}

pub(crate) fn ancestor_map(generations: usize) -> impl Fn(In<Entity>, Query<&ChildOf>) -> Option<Entity> {
    move |In(entity): In<Entity>, child_ofs: Query<&ChildOf>| get_ancestor(&child_ofs, entity, generations)
}

pub fn deref_copied<T: Deref>(x: T) -> T::Target
where
    <T as Deref>::Target: Copy,
{
    *x.deref()
}

pub fn deref_cloned<T: Deref>(x: T) -> T::Target
where
    <T as Deref>::Target: Clone,
{
    x.deref().clone()
}

/// Convenience trait for [`Send`] + [`Sync`] + 'static.
pub trait SSs: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> SSs for T {}
