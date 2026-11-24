//! Comparable counters and render-target snapshots from the guest command path.
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering::Relaxed}};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::HostList;

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VICTORY_VALIDATE").as_deref() == Ok("1"))
}

/// Record the exact host PSO for draws which the GPU confirms shaded pixels.
fn identify_shaded() -> bool {
    static IDENTIFY: OnceLock<bool> = OnceLock::new();
    *IDENTIFY.get_or_init(|| std::env::var("VICTORY_IDENTIFY_SHADED").as_deref() == Ok("1"))
}

static FIRST_BINDS: AtomicU64 = AtomicU64::new(0);
static READY_BEFORE_BIND: AtomicU64 = AtomicU64::new(0);
static COMPUTE_REQUESTED: AtomicU64 = AtomicU64::new(0);
static COMPUTE_DROPPED: AtomicU64 = AtomicU64::new(0);
static PENDING_MISSES: AtomicU64 = AtomicU64::new(0);
static FAILED_MISSES: AtomicU64 = AtomicU64::new(0);
static COMPLETED: AtomicU64 = AtomicU64::new(0);
static MEASURED: AtomicU64 = AtomicU64::new(0);
static ACTIVE: AtomicU64 = AtomicU64::new(0);
static SHADED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn first_bind(ready: bool) {
    FIRST_BINDS.fetch_add(1, Relaxed);
    if ready {
        READY_BEFORE_BIND.fetch_add(1, Relaxed);
    }
}

pub(crate) fn compute_pipeline_miss(pending: bool) {
    if pending { &PENDING_MISSES } else { &FAILED_MISSES }.fetch_add(1, Relaxed);
}

pub(crate) fn compute(ran: bool) {
    COMPUTE_REQUESTED.fetch_add(1, Relaxed);
    if !ran {
        COMPUTE_DROPPED.fetch_add(1, Relaxed);
    }
}

pub(crate) fn log(frame: u64, completed_fence: Option<u64>) {
    if enabled() {
        let checkpoint = if completed_fence.is_some() { "selected pass counters" } else { "render validation" };
        tracing::info!(frame, compute_requested = COMPUTE_REQUESTED.load(Relaxed),
            compute_dropped = COMPUTE_DROPPED.load(Relaxed), compute_pending = PENDING_MISSES.load(Relaxed),
            compute_suppressed_at_submit = crate::host::indexed_suppressed(true),
            graphics_suppressed_at_submit = crate::host::indexed_suppressed(false),
            compute_unavailable = FAILED_MISSES.load(Relaxed), first_binds = FIRST_BINDS.load(Relaxed),
            ready_before_bind = READY_BEFORE_BIND.load(Relaxed), draws_completed = COMPLETED.load(Relaxed),
            draws_measured = MEASURED.load(Relaxed), draws_active = ACTIVE.load(Relaxed),
            draws_shaded = SHADED.load(Relaxed), ?completed_fence, "{checkpoint}");
    }
}

const PAGE: u32 = 64;
type Storage = (ID3D12QueryHeap, ID3D12Resource);

#[derive(Default)]
pub(crate) struct Draws {
    count: u32,
    pages: Vec<Option<Storage>>,
    captures: Vec<crate::capture::PassImage>,
    selected: Vec<(u32, &'static str, u64)>,
    labels: Vec<Option<String>>,
}

impl Draws {
    pub(crate) fn reset(&mut self) {
        self.count = 0;
        self.captures.clear();
        self.selected.clear();
        self.labels.clear();
    }

    pub(crate) fn begin(list: &HostList) -> Option<u32> {
        if !enabled() { return None; }
        let mut d = list.validation.lock().unwrap_or_else(|e| e.into_inner());
        let index = d.count;
        d.count += 1;
        let page = (index / PAGE) as usize;
        if page == d.pages.len() {
            d.pages.push(crate::host::query_storage(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
                PAGE, size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>() as u64));
        }
        let (heap, _) = d.pages[page].as_ref()?;
        // SAFETY: this slot is unique until the command list is reset.
        unsafe { list.list.BeginQuery(heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, index % PAGE) };
        Some(index)
    }

    pub(crate) fn end(list: &HostList, query: Option<u32>, pipeline: crate::object::This, targets: &[usize], view_formats: &[i32]) {
        if !enabled() { return; }
        crate::capture::note_shader(pipeline);
        let mut d = list.validation.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = query {
            let (heap, _) = d.pages[(index / PAGE) as usize].as_ref().expect("started query");
            unsafe { list.list.EndQuery(heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, index % PAGE) };
            if identify_shaded() {
                // Keep an owned name: the title can release its pipeline after recording.
                d.labels.resize_with(index as usize + 1, || None);
                d.labels[index as usize] = Some(crate::pipeline::stage_names(pipeline));
            }
        }
        if let Some((hash, occurrence)) = capture_occurrence(pipeline) {
            if let Some(index) = query { d.selected.push((index, hash, occurrence)); }
            tracing::info!(hash, occurrence, frame = crate::queue::presented() + 1, ?view_formats, pipeline = %crate::pipeline::stage_names(pipeline), "matching comparison pass");
            for (slot, &target) in targets.iter().enumerate() {
                if let Some(image) = crate::capture::PassImage::record(list, target, hash, occurrence, slot, view_formats.get(slot).copied()) {
                    d.captures.push(image);
                }
            }
        }
    }

    pub(crate) fn resolve(&self, list: &ID3D12GraphicsCommandList) {
        for (page, storage) in self.pages.iter().take(self.count.div_ceil(PAGE) as usize).enumerate() {
            if let Some((heap, back)) = storage {
                let n = (self.count - page as u32 * PAGE).min(PAGE);
                unsafe { list.ResolveQueryData(heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, n, back, 0) };
            }
        }
    }

    /// True if this list contains a selected pass with a recorded query.
    pub(crate) fn completed(&mut self) -> bool {
        COMPLETED.fetch_add(u64::from(self.count), Relaxed);
        for (page, storage) in self.pages.iter().take(self.count.div_ceil(PAGE) as usize).enumerate() {
            let Some((_, back)) = storage else { continue };
            let n = (self.count - page as u32 * PAGE).min(PAGE);
            let mut data = std::ptr::null_mut();
            // The submission fence has completed and this page has n resolved queries.
            unsafe {
                if back.Map(0, None, Some(&mut data)).is_err() { continue; }
                let stats = std::slice::from_raw_parts(data.cast::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>(), n as usize);
                let first = page as u32 * PAGE;
                for &(index, hash, occurrence) in &self.selected {
                    if (first..first + n).contains(&index) {
                        let s = &stats[(index - first) as usize];
                        tracing::info!(hash, occurrence, input_vertices = s.IAVertices,
                            geometry_invocations = s.GSInvocations, geometry_primitives = s.GSPrimitives,
                            clipped_primitives = s.CPrimitives,
                            pixel_invocations = s.PSInvocations, "completed comparison pass");
                    }
                }
                let mut active = 0;
                let mut shaded = 0;
                for (slot, s) in stats.iter().enumerate() {
                    let draw = page * PAGE as usize + slot;
                    if s.PSInvocations != 0 && let Some(Some(label)) = self.labels.get(draw) {
                        tracing::info!(frame = crate::queue::presented() + 1, draw, pipeline = %label,
                            clipped_primitives = s.CPrimitives, pixel_invocations = s.PSInvocations,
                            "identified shaded draw");
                    }
                    active += u64::from(s.VSInvocations != 0 || s.CInvocations != 0 || s.PSInvocations != 0);
                    shaded += u64::from(s.PSInvocations != 0);
                }
                MEASURED.fetch_add(u64::from(n), Relaxed);
                ACTIVE.fetch_add(active, Relaxed);
                SHADED.fetch_add(shaded, Relaxed);
                back.Unmap(0, None);
            }
        }
        // A reusable command list keeps the buffers named by its copy commands
        // alive until reset. Save the first execution only.
        for image in &mut self.captures { image.save_once(); }
        !self.selected.is_empty()
    }
}

fn capture_occurrence(pipeline: crate::object::This) -> Option<(&'static str, u64)> {
    static PASSES: OnceLock<Vec<(String, AtomicU64)>> = OnceLock::new();
    static OCCURRENCES: OnceLock<Vec<u64>> = OnceLock::new();
    let passes = PASSES.get_or_init(|| crate::pipeline::parse_shader_hashes(
        &std::env::var("VICTORY_COMPARE_PASSES").unwrap_or_default()
    ).into_iter().map(|hash| (hash, AtomicU64::new(0))).collect());
    if passes.is_empty() { return None; }
    let (hash, count) = passes.iter().find(|(hash, _)| crate::pipeline::has_shader(pipeline, hash))?;
    let at = count.fetch_add(1, Relaxed) + 1;
    let selected = OCCURRENCES.get_or_init(|| std::env::var("VICTORY_COMPARE_OCCURRENCES")
        .unwrap_or_else(|_| "1,8,64".into()).split(',').filter_map(|s| s.trim().parse().ok()).collect());
    selected.contains(&at).then_some((hash.as_str(), at))
}
