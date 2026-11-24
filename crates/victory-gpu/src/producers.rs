//! Observe one exact indirect dispatch X argument across command-list recordings.
//! A reflected table UAV is a possible destination. Only a change bracketed
//! by GPU copies around a submitted command proves that its value changed.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::host::{BUFFER_READ, BUFFER_WRITE, HostList};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::Interface;

pub(crate) fn word() -> Option<u64> {
    static WORD: OnceLock<Option<u64>> = OnceLock::new();
    *WORD.get_or_init(|| {
        let value = std::env::var("VICTORY_PRODUCER_X").ok()?;
        let parsed = u64::from_str_radix(value.trim().trim_start_matches("0x"), 16)
            .ok()
            .filter(|a| a.is_multiple_of(4) && a.checked_add(3).is_some());
        if parsed.is_none() {
            tracing::warn!(
                value,
                "VICTORY_PRODUCER_X requires an aligned hexadecimal guest address"
            );
        }
        parsed
    })
}

pub(crate) fn covers(start: u64, bytes: u64, address: u64) -> bool {
    address
        .checked_sub(start)
        .and_then(|offset| offset.checked_add(4))
        .is_some_and(|end| end <= bytes)
}

pub(crate) fn record_index(base: u64, stride: u32, max: u32, address: u64) -> Option<u32> {
    let delta = address.checked_sub(base)?;
    if stride < 12 || !delta.is_multiple_of(u64::from(stride)) {
        return None;
    }
    let record = u32::try_from(delta / u64::from(stride)).ok()?;
    (record < max).then_some(record)
}

#[derive(Debug)]
enum Event {
    Candidate {
        shader: String,
        range: (u64, u64),
        status: String,
        buffer: Option<usize>,
        samples: bool,
    },
    Consumer {
        pipeline: String,
        kind: u32,
        record: u32,
        counted: bool,
        buffer: usize,
        samples: bool,
    },
}

const LIMIT: usize = 1024;
static NEXT_RECORDING: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct Recording {
    id: u64,
    events: Vec<Event>,
    omitted: u32,
}

#[derive(Default)]
pub(crate) struct Journal {
    records: Mutex<Recording>,
    back: OnceLock<Option<ID3D12Resource>>,
}

pub(crate) struct Sample {
    slot: usize,
    resource: ID3D12Resource,
    offset: u64,
}

impl Journal {
    pub(crate) fn reset(&self) {
        if word().is_none() {
            return;
        }
        let mut r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        r.events.clear();
        r.omitted = 0;
        r.id = 0;
    }

    fn add(&self, event: Event) -> Option<usize> {
        let mut r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        if r.events.len() == LIMIT {
            r.omitted = r.omitted.saturating_add(1);
            return None;
        }
        if r.id == 0 {
            r.id = NEXT_RECORDING.fetch_add(1, Ordering::Relaxed);
        }
        let index = r.events.len();
        r.events.push(event);
        Some(index)
    }

    fn storage(&self) -> Option<&ID3D12Resource> {
        self.back
            .get_or_init(|| crate::host::readback_buffer(8 * LIMIT as u64))
            .as_ref()
    }

    pub(crate) fn dropped(&self, shader: &str, range: (u64, u64), reason: String) {
        self.add(Event::Candidate {
            shader: shader.into(),
            range,
            status: reason,
            buffer: None,
            samples: false,
        });
    }

    pub(crate) fn before(
        list: &HostList,
        shader: &str,
        range: (u64, u64),
        address: u64,
    ) -> Option<Sample> {
        // The guest can name a UAV whose host view could not be made. Keep
        // the candidate, but never transition an upload-only copy to UAV.
        let shadow = crate::shadow::find(range.0, range.1).filter(|s| {
            // SAFETY: a resource retained by this shadow.
            let desc = unsafe { s.buffer.GetDesc() };
            desc.Flags.0
                & windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0
                != 0
        });
        Self::before_buffer(
            list,
            shader,
            range,
            shadow.as_ref().map(|s| (&s.buffer, s.offset(address))),
        )
    }

    fn before_buffer(
        list: &HostList,
        shader: &str,
        range: (u64, u64),
        resource: Option<(&ID3D12Resource, u64)>,
    ) -> Option<Sample> {
        let j = &list.producers;
        let slot = j.add(Event::Candidate {
            shader: shader.into(),
            range,
            status: "dispatch submitted".into(),
            buffer: resource.map(|(b, _)| b.as_raw() as usize),
            samples: resource.is_some() && j.storage().is_some(),
        })?;
        let (resource, offset) = resource?;
        let sample = Sample {
            slot,
            resource: resource.clone(),
            offset,
        };
        Self::copy(list, slot * 2, &sample.resource, sample.offset)?;
        // Binding already put this reflected UAV into the write state.
        list.buffer_state(&sample.resource, BUFFER_WRITE);
        Some(sample)
    }

    pub(crate) fn after(list: &HostList, sample: Option<Sample>) {
        if let Some(s) = sample {
            Self::copy(list, s.slot * 2 + 1, &s.resource, s.offset);
            list.buffer_state(&s.resource, BUFFER_WRITE);
        }
    }

    pub(crate) fn consumer(
        list: &HostList,
        pipeline: String,
        kind: u32,
        record: u32,
        args: (&ID3D12Resource, u64),
        count: Option<(&ID3D12Resource, u64)>,
    ) {
        let j = &list.producers;
        if let Some(slot) = j.add(Event::Consumer {
            pipeline,
            kind,
            record,
            counted: count.is_some(),
            buffer: args.0.as_raw() as usize,
            samples: j.storage().is_some(),
        }) {
            Self::copy(list, slot * 2, args.0, args.1);
            if let Some((buffer, at)) = count {
                Self::copy(list, slot * 2 + 1, buffer, at);
            }
        }
    }

    fn copy(list: &HostList, dword: usize, src: &ID3D12Resource, offset: u64) -> Option<()> {
        let back = list.producers.storage()?;
        list.buffer_state(src, BUFFER_READ);
        // SAFETY: callers checked that the four bytes belong to the source;
        // add() reserved the destination in this recording's readback.
        unsafe {
            list.list
                .CopyBufferRegion(back, dword as u64 * 4, src, offset, 4);
        }
        Some(())
    }

    pub(crate) fn report(&self, fence: u64, list_index: usize, unique_completed: bool) {
        let r = self.records.lock().unwrap_or_else(|e| e.into_inner());
        if r.events.is_empty() {
            return;
        }
        let address = word().expect("recorded when enabled");
        let back = self
            .back
            .get()
            .and_then(Option::as_ref)
            .filter(|_| unique_completed);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: the host completed this submission and the list was not
        // submitted twice into the same group (which reuses sample slots).
        let mapped = back
            .is_some_and(|b| unsafe { b.Map(0, None, Some(&mut ptr)) }.is_ok() && !ptr.is_null());
        for (order, event) in r.events.iter().enumerate() {
            let values = |sampled: bool, dword: usize| {
                (mapped && sampled)
                    .then(|| unsafe { ptr.cast::<u32>().add(order * 2 + dword).read() })
            };
            match event {
                Event::Candidate {
                    shader,
                    range,
                    status,
                    buffer,
                    samples,
                } => {
                    let (before, after) = (values(*samples, 0), values(*samples, 1));
                    let changed = before.zip(after).map(|(a, b)| a != b);
                    tracing::info!(fence, list_index, recording=r.id, order, address=format_args!("{address:#x}"),
                        shader, range=?range, status, ?buffer, ?before, ?after, ?changed, "indirect X producer candidate");
                }
                Event::Consumer {
                    pipeline,
                    kind,
                    record,
                    counted,
                    buffer,
                    samples,
                } => {
                    let x = values(*samples, 0);
                    let count = values(*samples && *counted, 1);
                    let selected = if *counted {
                        count.map(|n| *record < n)
                    } else {
                        Some(true)
                    };
                    tracing::info!(
                        fence,
                        list_index,
                        recording = r.id,
                        order,
                        address = format_args!("{address:#x}"),
                        pipeline,
                        kind,
                        record,
                        buffer,
                        ?x,
                        ?count,
                        ?selected,
                        "indirect X submitted consumer"
                    );
                }
            }
        }
        if mapped {
            unsafe { back.unwrap().Unmap(0, None) };
        }
        if r.omitted != 0 {
            tracing::warn!(
                fence,
                list_index,
                omitted = r.omitted,
                "indirect X trace limit reached"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_samples_distinguish_a_changed_word_from_a_bound_but_untouched_word() {
        use crate::host::host;
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;
        let Some(host) = host() else { return };
        let (buffer, ptr) = host.shared_buffer(16).expect("shared arguments");
        let (upload, value) = host.upload_buffer(4).expect("the value to copy");
        // SAFETY: these four test dwords belong to our live mapped buffers.
        unsafe {
            ptr.cast::<u32>().write(3);
            ptr.cast::<u32>().add(1).write(0);
            value.cast::<u32>().write(11);
        }
        let writer = host.new_list().unwrap();
        let reader = host.new_list().unwrap();
        // Record the reader first. It must observe the writer because the
        // lists are submitted writer-first on the same GPU queue.
        let probe = Journal::before_buffer(&reader, "untouched", (0, 16), Some((&buffer, 0)));
        Journal::after(&reader, probe);
        Journal::consumer(
            &reader,
            "mesh".into(),
            10,
            1,
            (&buffer, 0),
            Some((&buffer, 4)),
        );
        reader.close();

        let probe = Journal::before_buffer(&writer, "changed", (0, 16), Some((&buffer, 0)));
        writer.buffer_state(&buffer, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
        // SAFETY: one dword between live buffers on the open command list.
        unsafe { writer.list.CopyBufferRegion(&buffer, 0, &upload, 0, 4) };
        Journal::after(&writer, probe);
        writer.close();
        host.execute(&[&writer, &reader]);
        let read = |list: &HostList, n: usize| {
            let back = list.producers.storage().expect("probe readback");
            let mut p = std::ptr::null_mut();
            // SAFETY: the fence returned after both lists and their copies.
            unsafe {
                back.Map(0, None, Some(&mut p)).unwrap();
                let words = std::slice::from_raw_parts(p.cast::<u32>(), n).to_vec();
                back.Unmap(0, None);
                words
            }
        };
        assert_eq!(read(&writer, 2), [3, 11]);
        assert_eq!(read(&reader, 4), [11, 11, 11, 0]);
    }

    #[test]
    fn overlap_must_cover_the_whole_x_dword() {
        assert!(covers(0x1000, 12, 0x1008));
        assert!(!covers(0x1000, 11, 0x1008));
        assert!(!covers(0x1004, 8, 0x1000));
        assert!(!covers(u64::MAX, 1, u64::MAX));
    }
    #[test]
    fn only_active_record_x_positions_match() {
        assert_eq!(record_index(0x1000, 16, 3, 0x1020), Some(2));
        for a in [0x0ffc, 0x1004, 0x1008, 0x100c, 0x1030] {
            assert_eq!(record_index(0x1000, 16, 3, a), None);
        }
        assert_eq!(record_index(0, 0, 1, 0), None);
        assert_eq!(record_index(0, 12, 0, 0), None);
    }
}
