//! Fence-completed BatchCull groups and PackInstances counters. The mesh draw can
//! first expose its indirect X address after these dispatches were recorded.
mod pack;
pub(crate) use pack::{AfterPack, before_pack};

use super::{Event, covers};
use crate::host::{BUFFER_READ, HostList};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

type Range = (u64, u64);
pub(super) const BUDGET: u64 = 8 << 20;

#[derive(Copy, Clone, Debug)]
struct Layout {
    output: Range,
    stride: u32,
    kernel: Kernel,
}

#[derive(Copy, Clone, Debug)]
enum Kernel {
    Batch {
        group_base: u32,
        dispatched: u32,
        records: u32,
        table_words: u32,
    },
    Pack {
        elements: u32,
        start_word: u64,
        counter_words: u32,
    },
}

#[derive(Debug)]
pub(super) struct Input {
    back: ID3D12Resource,
    layout: Layout,
}

#[derive(Default)]
struct Waiting {
    inputs: std::collections::VecDeque<Snapshot>,
    bytes: usize,
    evicted: usize,
}

static WAITING: std::sync::Mutex<Waiting> = std::sync::Mutex::new(Waiting {
    inputs: std::collections::VecDeque::new(),
    bytes: 0,
    evicted: 0,
});

struct Snapshot {
    layout: Layout,
    words: Vec<u32>,
    fence: u64,
    list_index: usize,
    recording: u64,
    order: usize,
}

impl Input {
    pub(super) fn bytes(&self) -> u64 {
        match self.layout.kernel {
            Kernel::Batch {
                records,
                table_words,
                ..
            } => u64::from(records) * 16 + u64::from(table_words) * 4,
            Kernel::Pack {
                elements,
                counter_words,
                ..
            } => u64::from(elements) * 4 + u64::from(counter_words) * 8,
        }
    }

    pub(super) fn is_pack(&self) -> bool {
        matches!(self.layout.kernel, Kernel::Pack { .. })
    }

    fn snapshot(
        &self,
        fence: u64,
        list_index: usize,
        recording: u64,
        order: usize,
    ) -> Option<Snapshot> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: called only after the unique submission completed. The private
        // readback contains these bytes and stays mapped until they are copied.
        if unsafe { self.back.Map(0, None, Some(&mut ptr)) }.is_err() || ptr.is_null() {
            tracing::warn!(
                fence,
                list_index,
                recording,
                order,
                "could not map geometry compute inputs"
            );
            return None;
        }
        let words =
            unsafe { std::slice::from_raw_parts(ptr.cast::<u32>(), (self.bytes() / 4) as usize) }
                .to_vec();
        unsafe { self.back.Unmap(0, None) };
        Some(Snapshot {
            layout: self.layout,
            words,
            fence,
            list_index,
            recording,
            order,
        })
    }

    pub(super) fn complete(&self, fence: u64, list_index: usize, recording: u64, order: usize) {
        if super::word()
            .is_some_and(|x| output_record(self.layout.output, self.layout.stride, x).is_none())
        {
            return;
        }
        let Some(snap) = self.snapshot(fence, list_index, recording, order) else {
            return;
        };
        // Recheck under the same lock select() drains. This covers a different
        // CPU thread recording the target draw during the readback copy.
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(x) = super::word() {
            drop(waiting);
            snap.report(x);
        } else {
            waiting.push(snap);
        }
    }
}

impl Waiting {
    fn push(&mut self, snap: Snapshot) {
        let bytes = snap.words.len() * 4;
        while self.bytes + bytes > BUDGET as usize || self.inputs.len() >= 1024 {
            let Some(old) = self.inputs.pop_front() else {
                break;
            };
            self.bytes -= old.words.len() * 4;
            self.evicted += 1;
        }
        self.bytes += bytes;
        self.inputs.push_back(snap);
    }
}

/// Dispatches whose fences completed before the consumer was recorded retain
/// copied CPU words; command-list reset/replay cannot overwrite them.
pub(super) fn select(x: u64) {
    let pending = std::mem::take(&mut *WAITING.lock().unwrap_or_else(|e| e.into_inner()));
    if pending.evicted != 0 {
        tracing::warn!(
            evicted = pending.evicted,
            "geometry compute input history limit reached"
        );
    }
    for snap in pending.inputs {
        snap.report(x);
    }
}

impl Snapshot {
    fn batch_evidence(&self, records: u32, record: u32) -> Evidence {
        let (groups, table) = self.words.split_at(records as usize * 4);
        let mut hit_groups = Vec::new();
        let mut hits = 0;
        for (group, data) in groups.chunks_exact(4).enumerate() {
            let [a, b, c, d] = *data else { unreachable!() };
            if batch_index(b, c) == record {
                hits += 1;
                if hits <= 8 {
                    hit_groups.push((group as u32, [a, b, c, d]));
                }
            }
        }
        Evidence {
            hits,
            hit_groups,
            tile: table.get(record as usize).copied(),
        }
    }

    fn report(&self, address: u64) {
        let Some(record) = output_record(self.layout.output, self.layout.stride, address) else {
            return;
        };
        let (fence, list_index, recording, order) =
            (self.fence, self.list_index, self.recording, self.order);
        let Kernel::Batch {
            group_base,
            dispatched,
            records,
            ..
        } = self.layout.kernel
        else {
            self.report_pack(record);
            return;
        };
        let e = self.batch_evidence(records, record);
        for (group, [a, b, c, d]) in e.hit_groups {
            tracing::info!(fence, list_index, recording, order, record, group,
                source_record=u64::from(group_base)+u64::from(group),
                t82=?[a,b,c,d], mesh_slot=a & 0x1f_ffff,
                matrix_slot=b & 0x1f_ffff, cull_flags=(a>>21)&7,
                instance_base=d & 0xffff, "submitted BatchCull input group");
        }
        tracing::info!(fence, list_index, recording, order, record, hits=e.hits,
            dispatched, captured=records, t79=?e.tile,
            sentinel=e.tile.map(|v|v==u32::MAX), "submitted BatchCull input membership");
    }
}

struct Evidence {
    hits: u32,
    hit_groups: Vec<(u32, [u32; 4])>,
    tile: Option<u32>,
}

fn batch_index(word1: u32, word2: u32) -> u32 {
    ((word2 >> 24) << 10) | (word1 >> 22)
}

fn output_record((base, len): Range, stride: u32, x: u64) -> Option<u32> {
    if stride == 0 || !covers(base, len, x) {
        return None;
    }
    let delta = x - base;
    if !delta.is_multiple_of(u64::from(stride)) {
        return None;
    }
    u32::try_from(delta / u64::from(stride)).ok()
}

/// Copies the t82 records addressed by GroupID.x plus the root base, and
/// the t79 entries addressable by the output view. Never refresh title
/// memory here: bind() has already established the actual GPU inputs.
pub(crate) fn batch(
    list: &HostList,
    t82: Range,
    t79: Range,
    output: Range,
    base: u32,
    groups: u32,
    stride: u32,
) {
    if stride == 0 || groups == 0 || output.1 < 4 {
        return;
    }
    let first = u64::from(base) * 16;
    if first >= t82.1 {
        return;
    }
    let records = u64::from(groups).min((t82.1 - first) / 16);
    let table_words = (t79.1 / 4).min(1 + (output.1 - 4) / u64::from(stride));
    let Some(bytes) = (records * 16).checked_add(table_words * 4) else {
        return;
    };
    if records == 0 || table_words == 0 || bytes > BUDGET {
        return;
    }
    let Some(start) = t82.0.checked_add(first) else {
        return;
    };
    let Some(records_buffer) = crate::shadow::find(start, records * 16) else {
        return;
    };
    let Some(table_buffer) = crate::shadow::find(t79.0, table_words * 4) else {
        return;
    };
    let Some(record_state) = list.bound_buffer_state(&records_buffer.buffer) else {
        return;
    };
    let Some(table_state) = list.bound_buffer_state(&table_buffer.buffer) else {
        return;
    };
    let Some(back) = crate::host::readback_buffer(bytes) else {
        return;
    };
    let event = Input {
        back: back.clone(),
        layout: Layout {
            output,
            stride,
            kernel: Kernel::Batch {
                group_base: base,
                dispatched: groups,
                records: records as u32,
                table_words: table_words as u32,
            },
        },
    };
    if list.producers.add(Event::GeometryInput(event)).is_none() {
        return;
    }
    for (src, guest, len, dest, state) in [
        (records_buffer, start, records * 16, 0, record_state),
        (
            table_buffer,
            t79.0,
            table_words * 4,
            records * 16,
            table_state,
        ),
    ] {
        list.buffer_state(&src.buffer, BUFFER_READ);
        list.keep(&src.buffer);
        // SAFETY: each byte range is checked against its source view, shadow and destination.
        unsafe {
            list.list
                .CopyBufferRegion(&back, dest, &src.buffer, src.offset(guest), len)
        };
        list.buffer_state(&src.buffer, state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_group_and_output_record_use_different_units() {
        assert_eq!(batch_index((207 << 22) | 17, (1 << 24) | 244), 1231);
        assert_eq!(
            output_record((0x6615c00, 0x5ae0), 16, 0x661a8f0),
            Some(1231)
        );
        assert_eq!(output_record((0x6615c00, 0x5ae0), 16, 0x661a8f4), None);
        assert_eq!(output_record((8, 32), 16, 40), None);
    }

    #[test]
    fn fenced_batch_inputs_select_only_dispatched_groups_after_the_address_is_known() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut fixture = vec![0u32; 32 + 1232];
        let t79_word = 32;
        let expected = [(5 << 21) | 91, (207 << 22) | 73, 1 << 24, 0x0040_0011];
        // Groups outside the submitted [base=1, base+count=3) also match.
        for record in [0, 2, 3] {
            fixture[record * 4..record * 4 + 4].copy_from_slice(&expected);
        }
        fixture[t79_word + 1231] = 0xfedc_0001;
        let start = fixture.as_ptr() as u64;
        let len = fixture.len() as u64 * 4;
        let key = start as usize;
        crate::shadow::register(key, start, len);
        let shadow = crate::shadow::find(start, len).unwrap();
        let mut dst = std::ptr::null_mut();
        // SAFETY: write the test's GPU-visible shadow, before submission.
        unsafe {
            shadow.buffer.Map(0, None, Some(&mut dst)).unwrap();
            std::ptr::copy_nonoverlapping(
                fixture.as_ptr(),
                dst.cast::<u8>().add(shadow.offset(start) as usize).cast(),
                fixture.len(),
            );
            shadow.buffer.Unmap(0, None);
        }
        let list = host.new_list().unwrap();
        list.buffer_state(&shadow.buffer, BUFFER_READ);
        batch(
            &list,
            (start, 64),
            (start + t79_word as u64 * 4, 1232 * 4),
            (0x6615c00, 0x5ae0),
            1,
            2,
            16,
        );
        list.close();
        host.execute(&[&list]);
        let r = list
            .producers
            .records
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let [Event::GeometryInput(input)] = r.events.as_slice() else {
            panic!("missing GPU snapshot")
        };
        let mut pending = Waiting::default();
        pending.push(input.snapshot(1, 0, 1, 0).unwrap());
        let snap = pending.inputs.pop_front().unwrap();
        let record = output_record(snap.layout.output, snap.layout.stride, 0x661a8f0).unwrap();
        let hit = snap.batch_evidence(2, record);
        assert_eq!(hit.hits, 1);
        assert_eq!(hit.hit_groups, [(1, expected)]);
        assert_eq!(hit.tile, Some(0xfedc_0001));
        assert_eq!(snap.batch_evidence(2, 207).hits, 0);
        crate::shadow::unregister(key, start);
    }
}
