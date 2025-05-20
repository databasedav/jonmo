use bevy_ecs::prelude::*;
use bevy_reflect::{FromReflect, GetTypeRegistration, Reflect, Typed};
#[doc(no_inline)]
pub use enclose::enclose as clone;
use std::sync::{Arc, OnceLock};

#[derive(Default, Clone)]
pub struct LazyEntity(Arc<OnceLock<Entity>>);

impl LazyEntity {
    /// Creates a new `EntityHolder` with an empty `OnceLock`.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, entity: Entity) {
        // Set the entity in the OnceLock, panicking if it was already set
        self.0
            .set(entity)
            .expect("EntityHolder already contains an Entity");
    }

    /// Returns a reference to the `Entity` held by this `EntityHolder`.
    /// If the `Entity` is not set, it will panic.
    pub fn get(&self) -> Entity {
        self.0
            .get()
            .copied()
            .expect("EntityHolder does not contain an Entity")
    }
}

pub trait SSs: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> SSs for T {}

pub trait BigReflect: Reflect + FromReflect + GetTypeRegistration + Typed {}
impl<T: Reflect + FromReflect + GetTypeRegistration + Typed> BigReflect for T {}
