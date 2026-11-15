use alloc::{collections::BTreeMap, vec::Vec};
use bevy_ecs::prelude::Component;

use crate::{signal_map::MapDiff, signal_vec::VecDiff};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReplayRevision(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReplayRevisionOverflow;

impl ReplayRevision {
    pub(crate) const FIRST: Self = Self(1);

    pub(crate) fn get(self) -> u64 {
        self.0
    }

    fn next(self) -> Result<Self, ReplayRevisionOverflow> {
        self.0.checked_add(1).map(Self).ok_or(ReplayRevisionOverflow)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplayCacheReadiness {
    Warming,
    Ready { revision: ReplayRevision },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum ReplayCacheState<T> {
    #[default]
    Warming,
    Ready {
        current: T,
        revision: ReplayRevision,
    },
}

impl<T> ReplayCacheState<T> {
    pub(crate) fn ready(current: T) -> Self {
        Self::Ready {
            current,
            revision: ReplayRevision::FIRST,
        }
    }

    pub(crate) fn readiness(&self) -> ReplayCacheReadiness {
        match self {
            Self::Warming => ReplayCacheReadiness::Warming,
            Self::Ready { revision, .. } => ReplayCacheReadiness::Ready { revision: *revision },
        }
    }

    pub(crate) fn snapshot(&self) -> ReplayCacheSnapshot<'_, T> {
        match self {
            Self::Warming => ReplayCacheSnapshot::Warming,
            Self::Ready { current, revision } => ReplayCacheSnapshot::Ready {
                current,
                revision: *revision,
            },
        }
    }

    fn next_revision(&self) -> Result<ReplayRevision, ReplayRevisionOverflow> {
        match self {
            Self::Warming => Ok(ReplayRevision::FIRST),
            Self::Ready { revision, .. } => revision.next(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplayCacheSnapshot<'a, T> {
    Warming,
    Ready { current: &'a T, revision: ReplayRevision },
}

#[cfg(test)]
impl<'a, T> ReplayCacheSnapshot<'a, T> {
    pub(crate) fn cloned(self) -> ReplayCacheState<T>
    where
        T: Clone,
    {
        match self {
            Self::Warming => ReplayCacheState::Warming,
            Self::Ready { current, revision } => ReplayCacheState::Ready {
                current: current.clone(),
                revision,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReplayCacheUpdate<T> {
    Suppressed,
    Accepted { output: T, revision: ReplayRevision },
}

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScalarReplayCache<T> {
    state: ReplayCacheState<T>,
}

impl<T> Default for ScalarReplayCache<T> {
    fn default() -> Self {
        Self {
            state: ReplayCacheState::Warming,
        }
    }
}

impl<T> ScalarReplayCache<T> {
    pub(crate) fn warming() -> Self {
        Self::default()
    }

    pub(crate) fn ready(current: T) -> Self {
        Self {
            state: ReplayCacheState::ready(current),
        }
    }

    pub(crate) fn readiness(&self) -> ReplayCacheReadiness {
        self.state.readiness()
    }

    pub(crate) fn snapshot(&self) -> ReplayCacheSnapshot<'_, T> {
        self.state.snapshot()
    }

    pub(crate) fn accept(&mut self, output: T) -> Result<ReplayCacheUpdate<T>, ReplayRevisionOverflow>
    where
        T: Clone,
    {
        let revision = self.state.next_revision()?;
        self.state = ReplayCacheState::Ready {
            current: output.clone(),
            revision,
        };
        Ok(ReplayCacheUpdate::Accepted { output, revision })
    }
}

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub(crate) struct VecReplayCache<T> {
    state: ReplayCacheState<Vec<T>>,
}

impl<T> Default for VecReplayCache<T> {
    fn default() -> Self {
        Self {
            state: ReplayCacheState::Warming,
        }
    }
}

impl<T> VecReplayCache<T> {
    pub(crate) fn warming() -> Self {
        Self::default()
    }

    pub(crate) fn ready(current: Vec<T>) -> Self {
        Self {
            state: ReplayCacheState::ready(current),
        }
    }

    pub(crate) fn readiness(&self) -> ReplayCacheReadiness {
        self.state.readiness()
    }

    pub(crate) fn snapshot(&self) -> ReplayCacheSnapshot<'_, Vec<T>> {
        self.state.snapshot()
    }

    pub(crate) fn accept(
        &mut self,
        mut output: Vec<VecDiff<T>>,
    ) -> Result<ReplayCacheUpdate<Vec<VecDiff<T>>>, ReplayRevisionOverflow>
    where
        T: Clone,
    {
        let (first_accepted, initial_len) = match &self.state {
            ReplayCacheState::Warming => (
                output.iter().position(|diff| matches!(diff, VecDiff::Replace { .. })),
                0,
            ),
            ReplayCacheState::Ready { current, .. } => (Some(0), current.len()),
        };
        let Some(first_accepted) = first_accepted else {
            return Ok(ReplayCacheUpdate::Suppressed);
        };

        if first_accepted != 0 {
            output.drain(..first_accepted);
        }

        if !vec_diffs_are_valid(initial_len, &output) {
            return Ok(ReplayCacheUpdate::Suppressed);
        }

        let revision = self.state.next_revision()?;
        let start = output
            .iter()
            .rposition(|diff| matches!(diff, VecDiff::Replace { .. }))
            .unwrap_or(0);
        let current = match &mut self.state {
            ReplayCacheState::Warming => {
                self.state = ReplayCacheState::Ready {
                    current: Vec::new(),
                    revision,
                };
                let ReplayCacheState::Ready { current, .. } = &mut self.state else {
                    unreachable!("state was just set to Ready");
                };
                current
            }
            ReplayCacheState::Ready {
                current,
                revision: stored_revision,
            } => {
                *stored_revision = revision;
                current
            }
        };
        for diff in &output[start..] {
            let applied = apply_vec_diff(current, diff);
            debug_assert!(applied, "pre-validated vec diff batch failed to apply");
        }

        Ok(ReplayCacheUpdate::Accepted { output, revision })
    }
}

fn vec_diffs_are_valid<T>(mut len: usize, diffs: &[VecDiff<T>]) -> bool {
    for diff in diffs {
        match diff {
            VecDiff::Replace { values } => len = values.len(),
            VecDiff::InsertAt { index, .. } => {
                if *index > len {
                    return false;
                }
                len += 1;
            }
            VecDiff::UpdateAt { index, .. } => {
                if *index >= len {
                    return false;
                }
            }
            VecDiff::RemoveAt { index } => {
                if *index >= len {
                    return false;
                }
                len -= 1;
            }
            VecDiff::Move { old_index, new_index } => {
                if *old_index >= len || *new_index >= len {
                    return false;
                }
            }
            VecDiff::Push { .. } => len += 1,
            VecDiff::Pop => {
                if len == 0 {
                    return false;
                }
                len -= 1;
            }
            VecDiff::Clear => len = 0,
        }
    }
    true
}

fn apply_vec_diff<T: Clone>(current: &mut Vec<T>, diff: &VecDiff<T>) -> bool {
    match diff {
        VecDiff::Replace { values } => current.clone_from(values),
        VecDiff::InsertAt { index, value } => {
            if *index > current.len() {
                return false;
            }
            current.insert(*index, value.clone());
        }
        VecDiff::UpdateAt { index, value } => {
            let Some(current_value) = current.get_mut(*index) else {
                return false;
            };
            current_value.clone_from(value);
        }
        VecDiff::RemoveAt { index } => {
            if *index >= current.len() {
                return false;
            }
            current.remove(*index);
        }
        VecDiff::Move { old_index, new_index } => {
            if *old_index >= current.len() || *new_index >= current.len() {
                return false;
            }
            let value = current.remove(*old_index);
            current.insert(*new_index, value);
        }
        VecDiff::Push { value } => current.push(value.clone()),
        VecDiff::Pop => {
            if current.pop().is_none() {
                return false;
            }
        }
        VecDiff::Clear => current.clear(),
    }
    true
}

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MapReplayCache<K, V> {
    state: ReplayCacheState<BTreeMap<K, V>>,
}

impl<K, V> Default for MapReplayCache<K, V> {
    fn default() -> Self {
        Self {
            state: ReplayCacheState::Warming,
        }
    }
}

impl<K, V> MapReplayCache<K, V> {
    pub(crate) fn warming() -> Self {
        Self::default()
    }

    pub(crate) fn ready(current: BTreeMap<K, V>) -> Self {
        Self {
            state: ReplayCacheState::ready(current),
        }
    }

    pub(crate) fn readiness(&self) -> ReplayCacheReadiness {
        self.state.readiness()
    }

    pub(crate) fn snapshot(&self) -> ReplayCacheSnapshot<'_, BTreeMap<K, V>> {
        self.state.snapshot()
    }

    pub(crate) fn accept(
        &mut self,
        mut output: Vec<MapDiff<K, V>>,
    ) -> Result<ReplayCacheUpdate<Vec<MapDiff<K, V>>>, ReplayRevisionOverflow>
    where
        K: Clone + Ord,
        V: Clone,
    {
        let first_accepted = match &self.state {
            ReplayCacheState::Warming => output.iter().position(|diff| matches!(diff, MapDiff::Replace { .. })),
            ReplayCacheState::Ready { .. } => Some(0),
        };
        let Some(first_accepted) = first_accepted else {
            return Ok(ReplayCacheUpdate::Suppressed);
        };

        if first_accepted != 0 {
            output.drain(..first_accepted);
        }

        for diff in &mut output {
            if let MapDiff::Replace { entries } = diff {
                canonicalize_replace_entries(entries);
            }
        }

        if output.iter().any(|diff| matches!(diff, MapDiff::Update { .. })) {
            let keep = {
                let mut base = match &self.state {
                    ReplayCacheState::Warming => MapKeyBase::Empty,
                    ReplayCacheState::Ready { current, .. } => MapKeyBase::Map(current),
                };
                let mut overlay = BTreeMap::<&K, bool>::new();
                let mut keep = Vec::with_capacity(output.len());
                for diff in &output {
                    let should_keep = match diff {
                        MapDiff::Replace { entries } => {
                            base = MapKeyBase::Replaced(entries.as_slice());
                            overlay.clear();
                            true
                        }
                        MapDiff::Clear => {
                            base = MapKeyBase::Empty;
                            overlay.clear();
                            true
                        }
                        MapDiff::Insert { key, .. } => {
                            overlay.insert(key, true);
                            true
                        }
                        MapDiff::Remove { key } => {
                            overlay.insert(key, false);
                            true
                        }
                        MapDiff::Update { key, .. } => {
                            overlay.get(key).copied().unwrap_or_else(|| base.contains_key(key))
                        }
                    };
                    keep.push(should_keep);
                }
                keep
            };
            if keep.iter().any(|kept| !*kept) {
                let mut keep_iter = keep.into_iter();
                output.retain(|_| keep_iter.next().unwrap_or(true));
                if output.is_empty() {
                    return Ok(ReplayCacheUpdate::Suppressed);
                }
            }
        }

        let revision = self.state.next_revision()?;
        let start = output
            .iter()
            .rposition(|diff| matches!(diff, MapDiff::Replace { .. }))
            .unwrap_or(0);
        let current = match &mut self.state {
            ReplayCacheState::Warming => {
                self.state = ReplayCacheState::Ready {
                    current: BTreeMap::new(),
                    revision,
                };
                let ReplayCacheState::Ready { current, .. } = &mut self.state else {
                    unreachable!("state was just set to Ready");
                };
                current
            }
            ReplayCacheState::Ready {
                current,
                revision: stored_revision,
            } => {
                *stored_revision = revision;
                current
            }
        };
        for diff in &output[start..] {
            apply_map_diff(current, diff);
        }

        Ok(ReplayCacheUpdate::Accepted { output, revision })
    }
}

enum MapKeyBase<'a, K, V> {
    Empty,
    Map(&'a BTreeMap<K, V>),
    Replaced(&'a [(K, V)]),
}

impl<K: Ord, V> MapKeyBase<'_, K, V> {
    fn contains_key(&self, key: &K) -> bool {
        match self {
            Self::Empty => false,
            Self::Map(map) => map.contains_key(key),
            Self::Replaced(entries) => entries.binary_search_by(|(entry_key, _)| entry_key.cmp(key)).is_ok(),
        }
    }
}

fn canonicalize_replace_entries<K: Ord, V>(entries: &mut Vec<(K, V)>) {
    if !entries.windows(2).all(|pair| pair[0].0 < pair[1].0) {
        *entries = BTreeMap::from_iter(core::mem::take(entries)).into_iter().collect();
    }
}

fn apply_map_diff<K, V>(current: &mut BTreeMap<K, V>, diff: &MapDiff<K, V>)
where
    K: Clone + Ord,
    V: Clone,
{
    match diff {
        MapDiff::Replace { entries } => {
            *current = entries.iter().cloned().collect();
        }
        MapDiff::Insert { key, value } => {
            current.insert(key.clone(), value.clone());
        }
        MapDiff::Update { key, value } => {
            if let Some(slot) = current.get_mut(key) {
                slot.clone_from(value);
            }
        }
        MapDiff::Remove { key } => {
            current.remove(key);
        }
        MapDiff::Clear => current.clear(),
    }
}

#[cfg(test)]
mod tests {
    use alloc::{collections::BTreeMap, vec, vec::Vec};

    use super::*;

    fn ready_snapshot<'a, T>(snapshot: ReplayCacheSnapshot<'a, T>) -> (&'a T, ReplayRevision) {
        match snapshot {
            ReplayCacheSnapshot::Warming => panic!("expected ready replay cache"),
            ReplayCacheSnapshot::Ready { current, revision } => (current, revision),
        }
    }

    #[test]
    fn revision_overflow_is_checked_and_atomic_registration_ownership_observability() {
        let mut cache = ScalarReplayCache {
            state: ReplayCacheState::Ready {
                current: 10,
                revision: ReplayRevision(u64::MAX),
            },
        };

        assert_eq!(cache.accept(20), Err(ReplayRevisionOverflow));
        assert_eq!(
            cache.snapshot(),
            ReplayCacheSnapshot::Ready {
                current: &10,
                revision: ReplayRevision(u64::MAX),
            }
        );
    }

    #[test]
    fn scalar_accepts_every_output_and_ready_none_is_not_warming_registration_ownership_observability() {
        let mut cache = ScalarReplayCache::<Option<i32>>::warming();
        assert_eq!(cache.snapshot(), ReplayCacheSnapshot::Warming);

        assert!(matches!(
            cache.accept(None),
            Ok(ReplayCacheUpdate::Accepted {
                output: None,
                revision: ReplayRevision(1),
            })
        ));
        assert_eq!(
            cache.snapshot(),
            ReplayCacheSnapshot::Ready {
                current: &None,
                revision: ReplayRevision(1),
            }
        );

        assert!(matches!(
            cache.accept(None),
            Ok(ReplayCacheUpdate::Accepted {
                output: None,
                revision: ReplayRevision(2),
            })
        ));
    }

    #[test]
    fn vec_waits_for_replace_then_applies_ordered_suffix_registration_ownership_observability() {
        let mut cache = VecReplayCache::warming();

        assert!(matches!(
            cache.accept(vec![VecDiff::Push { value: 99 }, VecDiff::Clear]),
            Ok(ReplayCacheUpdate::Suppressed)
        ));
        assert_eq!(cache.snapshot(), ReplayCacheSnapshot::Warming);

        let update = cache
            .accept(vec![
                VecDiff::Push { value: 98 },
                VecDiff::Replace { values: vec![1, 2] },
                VecDiff::InsertAt { index: 1, value: 3 },
                VecDiff::Move {
                    old_index: 0,
                    new_index: 2,
                },
                VecDiff::Pop,
            ])
            .expect("the initial replay revision must be available");
        let ReplayCacheUpdate::Accepted { output, revision } = update else {
            panic!("a replacement must establish readiness");
        };
        assert_eq!(revision, ReplayRevision(1));
        assert_eq!(output.len(), 4, "the pre-baseline prefix must be discarded");

        let (current, observed_revision) = ready_snapshot(cache.snapshot());
        assert_eq!(current, &[3, 2]);
        assert_eq!(observed_revision, revision);
    }

    #[test]
    fn vec_empty_replace_is_ready_and_invalid_batches_are_atomic_registration_ownership_observability() {
        let mut cache = VecReplayCache::warming();
        cache
            .accept(vec![VecDiff::Replace {
                values: Vec::<i32>::new(),
            }])
            .expect("the initial replay revision must be available");
        assert!(ready_snapshot(cache.snapshot()).0.is_empty());

        assert!(matches!(
            cache.accept(vec![
                VecDiff::Push { value: 7 },
                VecDiff::InsertAt { index: 4, value: 1 },
                VecDiff::UpdateAt { index: 3, value: 2 },
                VecDiff::RemoveAt { index: 2 },
                VecDiff::Move {
                    old_index: 0,
                    new_index: 5,
                },
            ]),
            Ok(ReplayCacheUpdate::Suppressed)
        ));
        let (current, revision) = ready_snapshot(cache.snapshot());
        assert!(current.is_empty());
        assert_eq!(revision, ReplayRevision(1));
    }

    #[test]
    fn map_waits_for_replace_then_applies_ordered_suffix_registration_ownership_observability() {
        let mut cache = MapReplayCache::warming();

        assert!(matches!(
            cache.accept(vec![MapDiff::Insert { key: 9, value: 90 }, MapDiff::Clear]),
            Ok(ReplayCacheUpdate::Suppressed)
        ));

        let update = cache
            .accept(vec![
                MapDiff::Remove { key: 9 },
                MapDiff::Replace {
                    entries: vec![(1, 10), (2, 20)],
                },
                MapDiff::Update { key: 1, value: 11 },
                MapDiff::Insert { key: 3, value: 30 },
                MapDiff::Remove { key: 2 },
            ])
            .expect("the initial replay revision must be available");
        let ReplayCacheUpdate::Accepted { output, revision } = update else {
            panic!("a replacement must establish readiness");
        };
        assert_eq!(output.len(), 4, "the pre-baseline prefix must be discarded");
        assert_eq!(revision, ReplayRevision(1));

        let expected = BTreeMap::from([(1, 11), (3, 30)]);
        let (current, observed_revision) = ready_snapshot(cache.snapshot());
        assert_eq!(current, &expected);
        assert_eq!(observed_revision, revision);
    }

    #[test]
    fn map_empty_replace_and_snapshot_clone_preserve_one_state_observation_registration_ownership_observability() {
        let mut cache = MapReplayCache::warming();
        cache
            .accept(vec![MapDiff::Replace {
                entries: Vec::<(i32, i32)>::new(),
            }])
            .expect("the initial replay revision must be available");

        let cloned = cache.snapshot().cloned();
        assert_eq!(
            cloned,
            ReplayCacheState::Ready {
                current: BTreeMap::new(),
                revision: ReplayRevision(1),
            }
        );
    }

    #[test]
    fn replay_caches_are_graph_components_registration_ownership_observability() {
        fn assert_component<T: Component>() {}

        assert_component::<ScalarReplayCache<i32>>();
        assert_component::<VecReplayCache<i32>>();
        assert_component::<MapReplayCache<i32, i32>>();
    }

    #[test]
    fn ready_vec_and_map_caches_update_in_place_without_cloning_unchanged_items() {
        use bevy_platform::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Debug)]
        struct Counted {
            id: usize,
            clones: alloc::sync::Arc<AtomicUsize>,
        }

        impl PartialEq for Counted {
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }
        impl Eq for Counted {}
        impl PartialOrd for Counted {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Counted {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.id.cmp(&other.id)
            }
        }
        impl Clone for Counted {
            fn clone(&self) -> Self {
                self.clones.fetch_add(1, Ordering::Relaxed);
                Self {
                    id: self.id,
                    clones: self.clones.clone(),
                }
            }
        }

        let clones = alloc::sync::Arc::new(AtomicUsize::new(0));
        let item = |id| Counted {
            id,
            clones: clones.clone(),
        };

        let mut vec_cache = VecReplayCache::ready((0..64).map(&item).collect());
        assert_eq!(clones.load(Ordering::Relaxed), 0);
        vec_cache
            .accept(vec![VecDiff::Push { value: item(64) }])
            .expect("valid push");
        assert_eq!(
            clones.load(Ordering::Relaxed),
            1,
            "pushing one element into VecReplayCache must not clone existing elements"
        );

        clones.store(0, Ordering::Relaxed);
        let mut map_cache = MapReplayCache::ready((0..64).map(|i| (i, item(i))).collect());
        assert_eq!(clones.load(Ordering::Relaxed), 0);
        map_cache
            .accept(vec![MapDiff::Insert {
                key: 64,
                value: item(64),
            }])
            .expect("valid insert");
        assert_eq!(
            clones.load(Ordering::Relaxed),
            1,
            "inserting one entry into MapReplayCache must not clone existing entries"
        );
    }

    #[test]
    fn empty_warming_batches_are_suppressed_while_ready_empty_batches_pass_through() {
        let mut warming_vec = VecReplayCache::<i32>::warming();
        assert_eq!(warming_vec.accept(Vec::new()), Ok(ReplayCacheUpdate::Suppressed));

        let mut vec_cache = VecReplayCache::ready(vec![1, 2]);
        assert_eq!(
            vec_cache.accept(Vec::new()),
            Ok(ReplayCacheUpdate::Accepted {
                output: Vec::new(),
                revision: ReplayRevision(2),
            })
        );
        assert_eq!(ready_snapshot(vec_cache.snapshot()), (&vec![1, 2], ReplayRevision(2)));

        let mut warming_map = MapReplayCache::<i32, i32>::warming();
        assert_eq!(warming_map.accept(Vec::new()), Ok(ReplayCacheUpdate::Suppressed));

        let mut map_cache = MapReplayCache::ready(BTreeMap::from([(1, 10)]));
        assert_eq!(
            map_cache.accept(Vec::new()),
            Ok(ReplayCacheUpdate::Accepted {
                output: Vec::new(),
                revision: ReplayRevision(2),
            })
        );
        assert_eq!(
            ready_snapshot(map_cache.snapshot()),
            (&BTreeMap::from([(1, 10)]), ReplayRevision(2))
        );
    }

    #[test]
    fn map_replace_canonicalizes_unsorted_and_duplicate_entries_last_wins() {
        let mut cache = MapReplayCache::warming();
        let update = cache
            .accept(vec![MapDiff::Replace {
                entries: vec![(3, 30), (1, 10), (3, 300), (2, 20), (1, 100)],
            }])
            .expect("initial revision available");

        assert_eq!(
            update,
            ReplayCacheUpdate::Accepted {
                output: vec![MapDiff::Replace {
                    entries: vec![(1, 100), (2, 20), (3, 300)],
                }],
                revision: ReplayRevision(1),
            }
        );
        assert_eq!(
            ready_snapshot(cache.snapshot()),
            (&BTreeMap::from([(1, 100), (2, 20), (3, 300)]), ReplayRevision(1))
        );
    }

    #[test]
    fn map_absent_key_updates_are_filtered_and_purely_absent_batches_suppressed() {
        let mut cache = MapReplayCache::ready(BTreeMap::from([(1, 10), (2, 20)]));

        assert_eq!(
            cache.accept(vec![MapDiff::Update { key: 99, value: 990 }]),
            Ok(ReplayCacheUpdate::Suppressed)
        );
        assert_eq!(
            ready_snapshot(cache.snapshot()),
            (&BTreeMap::from([(1, 10), (2, 20)]), ReplayRevision(1))
        );

        let update = cache
            .accept(vec![
                MapDiff::Update { key: 99, value: 990 },
                MapDiff::Update { key: 1, value: 11 },
                MapDiff::Insert { key: 3, value: 30 },
                MapDiff::Update { key: 3, value: 33 },
                MapDiff::Remove { key: 2 },
                MapDiff::Update { key: 2, value: 22 },
            ])
            .expect("revision available");

        assert_eq!(
            update,
            ReplayCacheUpdate::Accepted {
                output: vec![
                    MapDiff::Update { key: 1, value: 11 },
                    MapDiff::Insert { key: 3, value: 30 },
                    MapDiff::Update { key: 3, value: 33 },
                    MapDiff::Remove { key: 2 },
                ],
                revision: ReplayRevision(2),
            }
        );
        assert_eq!(
            ready_snapshot(cache.snapshot()),
            (&BTreeMap::from([(1, 11), (3, 33)]), ReplayRevision(2))
        );

        let cleared = cache
            .accept(vec![MapDiff::Clear, MapDiff::Update { key: 1, value: 111 }])
            .expect("revision available");
        assert_eq!(
            cleared,
            ReplayCacheUpdate::Accepted {
                output: vec![MapDiff::Clear],
                revision: ReplayRevision(3),
            }
        );
        assert_eq!(ready_snapshot(cache.snapshot()), (&BTreeMap::new(), ReplayRevision(3)));
    }
}
