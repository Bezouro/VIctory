//! Late binding of heap SRVs selected by title constants copied at submission.
use super::*;

static INDEXED_SUPPRESSED: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn indexed_suppressed(compute: bool) -> u64 {
    INDEXED_SUPPRESSED[usize::from(compute)].load(Ordering::Relaxed)
}

pub(crate) struct CbvSrv {
    pub(crate) compute: bool,
    pub(crate) heap: crate::descriptor::RetainedHeap,
    pub(crate) upload: usize,
    pub(crate) read: victory_shader::resource::IndexedRead,
    #[cfg(feature = "diagnostics")]
    pub(crate) trace: Option<String>,
}

pub(crate) struct SrvPredicate {
    buffer: ID3D12Resource,
    offset: u64,
    pub(crate) mapped: usize,
}

impl SrvPredicate {
    pub(crate) fn run(&self, list: &HostList, command: impl FnOnce()) {
        // Guest predication is not implemented on these command paths.
        // Scope the predicate to the single command; nearby clears, copies
        // and dispatches must remain unconditional.
        unsafe {
            list.list
                .SetPredication(&self.buffer, self.offset, D3D12_PREDICATION_OP_EQUAL_ZERO);
        }
        command();
        unsafe {
            list.list
                .SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        }
    }
}

impl HostList {
    pub(crate) fn defer_indexed_srvs(&self, reads: Vec<CbvSrv>, predicate: &SrvPredicate) {
        let position = self
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .indexed_position();
        let mut deferred = self.deferred();
        deferred.has_indexed_srvs = true;
        deferred.copies.push(Step::IndexedSrvs {
            reads,
            position,
            predicate: predicate.mapped,
        });
    }

    pub(crate) fn srv_predicate(&self) -> Option<SrvPredicate> {
        self.scratch_slice(8, |buffer, _, offset, data| SrvPredicate {
            buffer: buffer.clone(),
            offset,
            mapped: data as usize,
        })
    }

    pub(super) fn prepare_indexed_srvs(
        &self,
        reads: &[CbvSrv],
        position: usize,
        predicate: usize,
        batch: &mut Batch,
    ) {
        let states = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let mut permitted = true;
        for selector in reads {
            let heap = selector.heap.get();
            let read = selector.read;
            let Some(host) = heap.host() else {
                permitted = false;
                continue;
            };
            // SAFETY: a dword in the list-owned upload after its Fill.
            let word = unsafe { (selector.upload as *const u32).read_unaligned() };
            let index = word.wrapping_shl(5).wrapping_add(read.byte_offset) >> 5;
            let (range, refusal) = if let Some(srv) = host.indexed_srv(heap, index) {
                let key = srv.shadow.buffer.as_raw() as usize;
                if !states.allows_indexed_read(key, position) {
                    (None, Some("buffer state at command"))
                } else if batch
                    .indexed_slots
                    .get(&(std::ptr::from_ref(heap) as usize, index))
                    .is_some_and(|previous| *previous != srv.words)
                {
                    (
                        None,
                        Some("descriptor changed while an earlier command still uses the slot"),
                    )
                } else if srv.materialize(host) {
                    batch
                        .indexed_slots
                        .insert((std::ptr::from_ref(heap) as usize, index), srv.words);
                    batch.add(&srv.shadow, srv.address as usize, srv.len as usize);
                    (Some((srv.address, srv.len)), None)
                } else {
                    (None, Some("stale buffer record"))
                }
            } else if (host.count()..host.guard_slot()).contains(&index) {
                (None, Some("host-private descriptor slot"))
            } else if host.stale_indexed_slot(heap, index) {
                (None, Some("stale indexed slot"))
            } else {
                (None, None) // empty slot, shader guard, or an unchanged other class
            };
            if let Some(reason) = refusal {
                permitted = false;
                static WARN: std::sync::Once = std::sync::Once::new();
                WARN.call_once(|| tracing::warn!(index, reason, "CBV-selected command suppressed"));
            }
            #[cfg(feature = "diagnostics")]
            if let Some(pipeline) = &selector.trace {
                tracing::info!(%pipeline, root_slot = read.slot, table_dwords = read.table_offset_dwords,
                                cbv_byte = read.dword * 4, index, ?range, ?refusal, "submitted CBV-selected buffer SRV");
            }
            let _ = range;
        }
        if !permitted {
            let compute = reads[0].compute;
            let count =
                INDEXED_SUPPRESSED[usize::from(compute)].fetch_add(1, Ordering::Relaxed) + 1;
            if count.is_power_of_two() {
                tracing::warn!(
                    compute,
                    count,
                    "CBV-selected commands suppressed at submission"
                );
            }
        }
        // SAFETY: aligned upload dword pair retained by this list.
        unsafe {
            (predicate as *mut u64).write(u64::from(permitted));
        }
    }
}

#[test]
fn late_srv_checks_the_state_at_the_dispatch_including_the_prelude() {
    let mut states = BufferStates::default();
    states.update(1, (), BUFFER_WRITE);
    let before = states.indexed_position();
    states.update(1, (), BUFFER_READ);
    let middle = states.indexed_position();
    states.update(1, (), BUFFER_WRITE);
    let after = states.indexed_position();
    states.update(2, (), BUFFER_WRITE); // first use: prelude applies to entire list
    assert!(!states.allows_indexed_read(1, before));
    assert!(states.allows_indexed_read(1, middle));
    assert!(!states.allows_indexed_read(1, after));
    assert!(!states.allows_indexed_read(2, middle));
    assert!(states.allows_indexed_read(3, middle)); // implicit read from COMMON
    states.clear();
    assert!(states.allows_indexed_read(1, states.sequence));
}
