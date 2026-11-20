//! Least-recently-used cache for kernel rows restricted to the active set.

use crate::kernel::Kernel;

const NIL: usize = usize::MAX;

struct Slot {
    key: usize,
    prev: usize,
    next: usize,
    /// number of restrictions applied to `row` (it is brought up to date when it is used)
    epoch: usize,
    row: Vec<f64>,
}

/// Distinct columns of the active set; `map[p]` is the index in `cols` of column `p`.
///
/// `cols` is empty if all columns are distinct (no mapping is needed then).
struct Dedup {
    cols: Vec<usize>,
    map: Vec<u32>,
}

/// LRU cache of kernel rows.
///
/// Rows are keyed by *kernel* index and contain the entries `K[key, cols[p]]` for the
/// current list of active columns `cols` (one column per active solver variable, so a
/// kernel index may occur several times, e.g. for ε-SVR). When the active set shrinks,
/// the cached rows are compacted instead of being recomputed; this happens lazily when a row
/// is used again, since most cached rows are not touched between two restrictions.
pub(crate) struct RowCache {
    slots: Vec<Slot>,
    /// `lookup[key]` = slot holding the row of `key`, or `NIL`
    lookup: Vec<usize>,
    head: usize,
    tail: usize,
    capacity: usize,
    cols: Vec<usize>,
    /// `history[e]`: the columns (of the layout after `e` restrictions) kept by restriction `e`
    history: Vec<Vec<u32>>,
    /// Lazily computed duplicate structure of `cols` (invalidated whenever `cols` changes).
    dedup: Option<Dedup>,
    /// Scratch buffer for the entries of the distinct columns.
    scratch: Vec<f64>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
}

impl RowCache {
    /// Creates a cache for a kernel with `n_kernel` samples and initial columns `cols`
    /// that keeps at most `capacity` rows (at least two).
    pub(crate) fn new(n_kernel: usize, cols: Vec<usize>, capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            lookup: vec![NIL; n_kernel],
            head: NIL,
            tail: NIL,
            capacity: capacity.max(2),
            cols,
            history: Vec::new(),
            dedup: None,
            scratch: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Capacity in rows for a memory budget given in megabytes.
    pub(crate) fn capacity_for(megabytes: f64, row_len: usize) -> usize {
        let bytes = (megabytes.max(0.0) * 1024.0 * 1024.0) as usize;
        bytes / (8 * row_len.max(1))
    }

    fn unlink(&mut self, s: usize) {
        let (prev, next) = (self.slots[s].prev, self.slots[s].next);
        if prev != NIL {
            self.slots[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.slots[next].prev = prev;
        } else {
            self.tail = prev;
        }
    }

    fn push_front(&mut self, s: usize) {
        self.slots[s].prev = NIL;
        self.slots[s].next = self.head;
        if self.head != NIL {
            self.slots[self.head].prev = s;
        }
        self.head = s;
        if self.tail == NIL {
            self.tail = s;
        }
    }

    /// Makes sure that the row of kernel index `key` is cached and returns its slot.
    ///
    /// The returned slot stays valid until the next call that inserts a row while the
    /// cache is full and this slot is the least recently used one. Because a slot is
    /// marked as most recently used here, two consecutive calls always yield two valid slots.
    pub(crate) fn ensure<K: Kernel + ?Sized>(&mut self, kernel: &K, key: usize) -> usize {
        let s = self.lookup[key];
        if s != NIL {
            self.hits += 1;
            self.refresh(s);
            if self.head != s {
                self.unlink(s);
                self.push_front(s);
            }
            return s;
        }
        self.misses += 1;
        let s = if self.slots.len() < self.capacity {
            self.slots.push(Slot {
                key,
                prev: NIL,
                next: NIL,
                epoch: self.history.len(),
                row: vec![0.0; self.cols.len()],
            });
            self.slots.len() - 1
        } else {
            let s = self.tail;
            self.unlink(s);
            self.lookup[self.slots[s].key] = NIL;
            self.slots[s].key = key;
            self.slots[s].epoch = self.history.len();
            self.slots[s].row.resize(self.cols.len(), 0.0);
            s
        };
        self.compute_row(kernel, key, s);
        self.lookup[key] = s;
        self.push_front(s);
        s
    }

    /// Finds the distinct columns (a kernel index may be used by several variables, e.g. the two
    /// copies of an ε-SVR sample) so that each kernel entry is evaluated only once.
    fn build_dedup(&self) -> Dedup {
        let mut first = vec![u32::MAX; self.lookup.len()];
        let mut cols = Vec::new();
        let mut map = Vec::with_capacity(self.cols.len());
        for &c in &self.cols {
            if first[c] == u32::MAX {
                first[c] = cols.len() as u32;
                cols.push(c);
            }
            map.push(first[c]);
        }
        if cols.len() == self.cols.len() {
            cols.clear();
            map.clear();
        }
        Dedup { cols, map }
    }

    /// Computes the row of `key` into slot `s`.
    fn compute_row<K: Kernel + ?Sized>(&mut self, kernel: &K, key: usize, s: usize) {
        if self.dedup.is_none() {
            self.dedup = Some(self.build_dedup());
        }
        let dedup = self.dedup.as_ref().expect("computed above");
        if dedup.map.is_empty() {
            kernel.row_into(key, &self.cols, &mut self.slots[s].row);
        } else {
            self.scratch.resize(dedup.cols.len(), 0.0);
            kernel.row_into(key, &dedup.cols, &mut self.scratch);
            for (o, &u) in self.slots[s].row.iter_mut().zip(&dedup.map) {
                *o = self.scratch[u as usize];
            }
        }
    }

    /// Applies the restrictions that happened since the row in slot `s` was last used.
    fn refresh(&mut self, s: usize) {
        let slot = &mut self.slots[s];
        for kept in &self.history[slot.epoch..] {
            for (w, &p) in kept.iter().enumerate() {
                slot.row[w] = slot.row[p as usize];
            }
            slot.row.truncate(kept.len());
        }
        slot.epoch = self.history.len();
    }

    /// Cached row stored in slot `s` (see [`RowCache::ensure`]).
    #[inline]
    pub(crate) fn row(&self, s: usize) -> &[f64] {
        &self.slots[s].row
    }

    /// Keeps only the columns `p` with `keep[p] == true` in all cached rows.
    pub(crate) fn restrict(&mut self, keep: &[bool]) {
        debug_assert_eq!(keep.len(), self.cols.len());
        let kept: Vec<u32> = (0..keep.len() as u32)
            .filter(|&p| keep[p as usize])
            .collect();
        for (w, &p) in kept.iter().enumerate() {
            self.cols[w] = self.cols[p as usize];
        }
        self.cols.truncate(kept.len());
        self.dedup = None;
        self.history.push(kept);
    }

    /// Drops all cached rows and switches to new columns.
    pub(crate) fn reset(&mut self, cols: Vec<usize>) {
        for slot in &self.slots {
            self.lookup[slot.key] = NIL;
        }
        self.slots.clear();
        self.history.clear();
        self.head = NIL;
        self.tail = NIL;
        self.cols = cols;
        self.dedup = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::PrecomputedKernel;

    fn matrix(n: usize) -> Vec<f64> {
        (0..n * n).map(|v| v as f64).collect()
    }

    #[test]
    fn evicts_least_recently_used() {
        let m = matrix(4);
        let k = PrecomputedKernel::new(&m, 4);
        let mut c = RowCache::new(4, vec![0, 1, 2, 3], 2);
        let s0 = c.ensure(&k, 0);
        assert_eq!(c.row(s0), &[0.0, 1.0, 2.0, 3.0]);
        c.ensure(&k, 1);
        c.ensure(&k, 0); // 0 is now most recent
        let s2 = c.ensure(&k, 2); // evicts 1
        assert_eq!(c.row(s2), &[8.0, 9.0, 10.0, 11.0]);
        assert_eq!((c.hits, c.misses), (1, 3));
        c.ensure(&k, 0);
        assert_eq!(c.hits, 2);
        c.ensure(&k, 1);
        assert_eq!(c.misses, 4);
    }

    #[test]
    fn restrict_and_reset() {
        let m = matrix(3);
        let k = PrecomputedKernel::new(&m, 3);
        let mut c = RowCache::new(3, vec![0, 1, 2, 0], 10);
        let s = c.ensure(&k, 1);
        assert_eq!(c.row(s), &[3.0, 4.0, 5.0, 3.0]);
        c.restrict(&[true, false, true, false]);
        let s = c.ensure(&k, 1);
        assert_eq!(c.row(s), &[3.0, 5.0]);
        let s = c.ensure(&k, 2);
        assert_eq!(c.row(s), &[6.0, 8.0]);
        c.reset(vec![2, 1]);
        let s = c.ensure(&k, 0);
        assert_eq!(c.row(s), &[2.0, 1.0]);
        assert_eq!(c.misses, 3);
    }

    #[test]
    fn duplicate_columns_are_evaluated_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counting<'a>(PrecomputedKernel<'a>, AtomicUsize);
        impl Kernel for Counting<'_> {
            fn len(&self) -> usize {
                self.0.len()
            }
            fn eval(&self, i: usize, j: usize) -> f64 {
                self.1.fetch_add(1, Ordering::Relaxed);
                self.0.eval(i, j)
            }
        }
        let m = matrix(3);
        let k = Counting(PrecomputedKernel::new(&m, 3), AtomicUsize::new(0));
        let mut c = RowCache::new(3, vec![0, 1, 2, 2, 1, 0], 10);
        let s = c.ensure(&k, 1);
        assert_eq!(c.row(s), &[3.0, 4.0, 5.0, 5.0, 4.0, 3.0]);
        assert_eq!(k.1.load(Ordering::Relaxed), 3);
        c.restrict(&[true, true, false, true, false, false]);
        let s = c.ensure(&k, 2);
        assert_eq!(c.row(s), &[6.0, 7.0, 8.0]);
        assert_eq!(k.1.load(Ordering::Relaxed), 6);
        let s = c.ensure(&k, 1);
        assert_eq!(c.row(s), &[3.0, 4.0, 5.0]);
    }
}
