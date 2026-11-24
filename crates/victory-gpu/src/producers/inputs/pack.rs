//! Capture exactly the counters and indices used by CS_PackInstances.
use super::{Event, Input, Kernel, Layout, Range, Snapshot, output_record};
use crate::host::{BUFFER_READ, HostList};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

#[derive(Debug, PartialEq, Eq)]
struct PackEvidence {
    before: Option<u32>,
    after: Option<u32>,
    hits: usize,
    // t83 positions and their top-three-bit exponents (up to eight examples).
    entries: Vec<(usize, u32)>,
}

impl Snapshot {
    fn pack_evidence(
        &self,
        record: u32,
        elements: u32,
        start_word: u64,
        counter_words: u32,
    ) -> PackEvidence {
        let (ids, counters) = self.words.split_at(elements as usize);
        let (pre, post) = counters.split_at(counter_words as usize);
        let mut hits = 0;
        let mut entries = Vec::new();
        for (position, &packed) in ids.iter().enumerate() {
            if packed & 0x1fff_ffff == record {
                hits += 1;
                if entries.len() < 8 {
                    entries.push((position, packed >> 29));
                }
            }
        }
        let byte = u64::from(record) * u64::from(self.layout.stride);
        let index = (byte / 4)
            .checked_sub(start_word)
            .and_then(|i| usize::try_from(i).ok());
        PackEvidence {
            before: index.and_then(|i| pre.get(i).copied()),
            after: index.and_then(|i| post.get(i).copied()),
            hits,
            entries,
        }
    }

    pub(super) fn report_pack(&self, record: u32) {
        let Kernel::Pack {
            elements,
            start_word,
            counter_words,
        } = self.layout.kernel
        else {
            return;
        };
        let e = self.pack_evidence(record, elements, start_word, counter_words);
        let expected = match (e.before, e.entries.as_slice(), e.hits) {
            (Some(n), [(_, shift)], 1) => Some(n.min(0x1ffff).div_ceil(1 << shift)),
            _ => None,
        };
        tracing::info!(fence=self.fence,list_index=self.list_index,recording=self.recording,order=self.order,record,elements,
            hits=?Some(e.hits),positions_exponents=?e.entries,before=?e.before,after=?e.after,?expected,
            "submitted PackInstances input membership");
    }
}

pub(crate) struct AfterPack {
    src: ID3D12Resource,
    back: ID3D12Resource,
    offset: u64,
    length: u64,
    destination: u64,
    state: u32,
}

impl AfterPack {
    pub(crate) fn after(&self, list: &HostList) {
        self.copy_at(list, self.destination);
    }

    fn copy_at(&self, list: &HostList, destination: u64) {
        list.buffer_state(&self.src, BUFFER_READ);
        // SAFETY: both counter snapshots use the same validated u8 extent.
        unsafe {
            list.list
                .CopyBufferRegion(&self.back, destination, &self.src, self.offset, self.length)
        };
        list.buffer_state(&self.src, self.state);
    }
}

pub(crate) fn before_pack(
    list: &HostList,
    t83: Range,
    elements: u32,
    output: Range,
    stride: u32,
    x: Option<u64>,
) -> Option<AfterPack> {
    let ids_bytes = u64::from(elements) * 4;
    if elements == 0 || elements > 16384 || ids_bytes > t83.1 || stride == 0 {
        return None;
    }
    let ids = crate::shadow::find(t83.0, ids_bytes)?;
    let ids_state = list.bound_buffer_state(&ids.buffer)?;
    // Until X is known, snapshot the u8 view if it fits the per-dispatch
    // limit. For larger views keep t83 membership; once X is known a
    // single dword suffices irrespective of view size.
    let counter_range = match x {
        Some(x) if output_record(output, stride, x).is_some() => Some((x, 4)),
        None if output.1 <= 512 << 10 && output.1 >= 4 => Some((output.0, output.1 & !3)),
        _ => None,
    };
    let counters = counter_range.and_then(|(start, len)| {
        let shadow = crate::shadow::find(start, len)?;
        let state = list.bound_buffer_state(&shadow.buffer)?;
        Some((start, len, shadow, state))
    });
    let (start_word, length) = counters
        .as_ref()
        .map_or((0, 0), |(start, len, _, _)| ((start - output.0) / 4, *len));
    let back = crate::host::readback_buffer(ids_bytes + 2 * length)?;
    let input = Input {
        back: back.clone(),
        layout: Layout {
            output,
            stride,
            kernel: Kernel::Pack {
                elements,
                start_word,
                counter_words: (length / 4) as u32,
            },
        },
    };
    list.producers.add(Event::GeometryInput(input))?;
    list.buffer_state(&ids.buffer, BUFFER_READ);
    list.keep(&ids.buffer);
    // SAFETY: t83 has elements dwords and the private readback starts with that span.
    unsafe {
        list.list
            .CopyBufferRegion(&back, 0, &ids.buffer, ids.offset(t83.0), ids_bytes)
    };
    list.buffer_state(&ids.buffer, ids_state);
    let (start, length, src, state) = counters?;
    let probe = AfterPack {
        src: src.buffer.clone(),
        back,
        offset: src.offset(start),
        length,
        destination: ids_bytes + length,
        state,
    };
    list.keep(&probe.src);
    probe.copy_at(list, ids_bytes);
    Some(probe)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn submitted_pack_copies_the_own_counter_on_both_sides_and_reports_the_t83_lane() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;
        let Some(host) = crate::host::host() else {
            return;
        };
        let fixture = [0u32; 32];
        let base = fixture.as_ptr() as u64;
        let key = base as usize;
        let len = std::mem::size_of_val(&fixture) as u64;
        crate::shadow::register(key, base, len);
        let shadow = crate::shadow::find(base, len).unwrap();
        let mut mapped = std::ptr::null_mut();
        let mut data = [0x5a5a; 32];
        data[..4].copy_from_slice(&[(1 << 29) | 1, 4, (2 << 29) | 3, (7 << 29) | 3]);
        // u8 begins at dword 8. Its record 3 starts 12 dwords later.
        data[20] = 17;
        unsafe {
            shadow.buffer.Map(0, None, Some(&mut mapped)).unwrap();
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                mapped.cast::<u8>().add(shadow.offset(base) as usize).cast(),
                data.len(),
            );
            shadow.buffer.Unmap(0, None);
        }
        let (upload, new_value) = host.upload_buffer(4).unwrap();
        unsafe { new_value.cast::<u32>().write(5) };
        let list = host.new_list().unwrap();
        list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
        // The fourth entry in the descriptor is outside the shader's active count.
        let after = before_pack(&list, (base, 16), 3, (base + 32, 80), 16, None).unwrap();
        assert_eq!(
            list.bound_buffer_state(&shadow.buffer),
            Some(crate::host::BUFFER_WRITE)
        );
        list.buffer_state(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
        // Positive control: one GPU writer updates only u8 record 3.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, shadow.offset(base + 80), &upload, 0, 4)
        };
        list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
        after.after(&list);
        assert_eq!(
            list.bound_buffer_state(&shadow.buffer),
            Some(crate::host::BUFFER_WRITE)
        );
        list.close();
        host.execute(&[&list]);
        let journal = list
            .producers
            .records
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let [Event::GeometryInput(input)] = journal.events.as_slice() else {
            panic!("missing pack snapshot")
        };
        let snap = input.snapshot(1, 0, 1, 0).unwrap();
        let Kernel::Pack {
            elements,
            start_word,
            counter_words,
        } = snap.layout.kernel
        else {
            panic!("wrong capture kind")
        };
        let record = output_record(snap.layout.output, snap.layout.stride, base + 80).unwrap();
        let matched = snap.pack_evidence(record, elements, start_word, counter_words);
        assert_eq!(
            matched,
            PackEvidence {
                before: Some(17),
                after: Some(5),
                hits: 1,
                entries: vec![(2, 2)]
            }
        );
        let neighbor = snap.pack_evidence(2, elements, start_word, counter_words);
        assert_eq!(
            neighbor,
            PackEvidence {
                before: Some(0x5a5a),
                after: Some(0x5a5a),
                hits: 0,
                entries: vec![]
            }
        );
        crate::shadow::unregister(key, base);
    }
}
