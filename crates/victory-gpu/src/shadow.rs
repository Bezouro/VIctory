//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Disjoint snapshots by offset, covering only declared shader writes.
    before: Mutex<BTreeMap<usize, Vec<u8>>>,
}

#[cfg(test)]
thread_local! {
    /// Tests: runs in copy-back between its read of title memory and its
    /// write, as a title thread's write would.
    static AFTER_READ: std::cell::RefCell<Option<Box<dyn Fn()>>> = const { std::cell::RefCell::new(None) };
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// only newly declared intervals need storage. World draws often add
    /// overlapping views of the same 174 MB heap to one submission.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        // The parts no earlier range of the submission declared: their
        // refresh was free to bring the title's bytes into the copy (a
        // refresh leaves declared ranges alone), and may just have.
        let fresh = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let declared = pending.iter().map(|&(a, e)| self.inside(a, e - a));
            let fresh = outside(from, to, declared.collect());
            pending.push((address, address + len));
            fresh
        };
        if fresh.is_empty() {
            return;
        }
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        for (start, end) in fresh {
            // SAFETY: these offsets are inside the mapped host buffer.
            let bytes = unsafe { std::slice::from_raw_parts(self.data.add(start), end - start) };
            // Keep existing chunks in place. A small addition at either end
            // of a large range must not copy the entire previous snapshot.
            let mut chunk = pooled(bytes.len());
            chunk.extend_from_slice(bytes);
            before.insert(start, chunk);
        }
    }

    /// The title-side clear has already supplied these bytes. Compare
    /// final GPU data to the latest clear where one ran, so a shader can
    /// restore the original value and still have its write copied back.
    pub(crate) fn expect_cleared(&self, address: usize, len: usize, pattern: &[u8]) {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (from, to) = self.inside(address, len);
        let first = before.range(..=from).next_back().map_or(from, |(&at, _)| at);
        for (&at, bytes) in before.range_mut(first..to) {
            let start = from.max(at);
            let end = to.min(at + bytes.len());
            if start < end {
                let phase = (start - from) % pattern.len();
                let dst = &mut bytes[start - at..end - at];
                let first = dst.len().min(pattern.len());
                for (i, byte) in dst[..first].iter_mut().enumerate() {
                    *byte = pattern[(phase + i) % pattern.len()];
                }
                let mut done = first;
                while done < dst.len() {
                    let n = done.min(dst.len() - done);
                    dst.copy_within(..n, done);
                    done += n;
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        if from < to {
            // A chunk can start before the requested range and overlap it.
            let first = snapshot.range(..=from).next_back().map_or(from, |(&at, _)| at);
            for (&at, bytes) in snapshot.range_mut(first..to) {
                let a = from.max(at);
                let b = to.min(at + bytes.len());
                if a < b {
                    written += self.merge_chunk(a, &mut bytes[a - at..b - at]);
                }
            }
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty() {
            for (_, bytes) in std::mem::take(&mut *snapshot) {
                give_back(bytes);
            }
        }
        written
    }

    /// Merge GPU changes in one captured interval back into guest memory.
    fn merge_chunk(&self, from: usize, before: &mut [u8]) -> usize {
        let start = self.start as usize;
        let to = from + before.len();
        let mut written = 0;
        // By 64 KiB piece: the copy compared with its bytes before the
        // submission, title memory read only where the GPU wrote.
        const PIECE: usize = 64 << 10;
        let mut title = vec![0u8; PIECE.min(to - from)];
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut at = from;
        while at < to {
            let n = (PIECE - (start + at) % PIECE).min(to - at);
            // SAFETY: inside the mapping; only the submitting thread
            // writes it.
            let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
            let old = &mut before[at - from..at - from + n];
            let title = &mut title[..n];
            if host != old {
                // Title threads can write other bytes after this read,
                // including through an alias. Refresh the page on the
                // next draw even when every GPU byte was written back.
                copied[(start + at) / PAGE - start / PAGE] = 0;
                if !crate::host::read_into((start + at) as *const u8, title) {
                    at += n;
                    continue;
                }
                // Title memory with the GPU's bytes put in; the copy
                // takes the title's bytes in turn, and so does the
                // snapshot: to a later range that overlaps this one,
                // they are not a shader's write to put back.
                #[cfg(test)]
                AFTER_READ.with(|hook| {
                    if let Some(f) = hook.borrow().as_ref() { f(); }
                });
                runs.clear();
                for (i, ((t, h), o)) in title.iter_mut().zip(host.iter_mut()).zip(old).enumerate() {
                    if *h != *o && *t != *h {
                        *t = *h;
                        match runs.last_mut() {
                            Some((_, end)) if *end == i => *end += 1,
                            _ => runs.push((i, i + 1)),
                        }
                    }
                    *h = *t;
                    *o = *t;
                }
                if let (Some(&(first, _)), Some(&(_, last))) = (runs.first(), runs.last()) {
                    crate::watch::writing(start + at + first, last - first);
                    written += crate::host::write_changed(start + at, title, &runs);
                }
            }
            at += n;
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// The parts of `[from, to)` outside every range in `ranges`, in order.
fn outside(from: usize, to: usize, mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let (mut at, mut parts) = (from, Vec::new());
    for (a, e) in ranges {
        if at < to && a > at {
            parts.push((at, a.min(to)));
        }
        at = at.max(e);
    }
    if at < to {
        parts.push((at, to));
    }
    parts
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// Existing copies of any of this range, including loose address views.
pub(crate) fn overlapping(from: u64, to: u64, mut visit: impl FnMut(&Arc<Shadow>, u64, u64)) {
    let mut intersect = |shadow: &Arc<Shadow>| {
        let start = from.max(shadow.start);
        let end = to.min(shadow.start + shadow.len);
        if start < end {
            visit(shadow, start, end);
        }
    };
    for (_, entry) in buffers().range(..(to, 0)).filter(|(_, e)| e.end > from) {
        if let Some(shadow) = &entry.shadow {
            intersect(shadow);
        }
    }
    for (_, shadow) in LOOSE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .range(..(to, 0))
    {
        intersect(shadow);
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    find_inner(address, len, true)
}

/// A shadow for a declared title buffer that contains the whole range.
pub(crate) fn find_registered(address: u64, len: u64) -> Option<Arc<Shadow>> {
    find_inner(address, len, false)
}

fn find_inner(address: u64, len: u64, allow_loose: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return if allow_loose {
            loose(address, end)
        } else {
            None
        };
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three pages of title memory on page bounds (plain heap memory, so
    /// not write-watched: refreshes copy whole ranges) and their host
    /// copy; `None` without a host GPU. Returns the memory's owner, the
    /// address of the first page and the copy.
    fn title() -> Option<(Vec<u8>, usize, Shadow)> {
        let memory = vec![0u8; 4 * PAGE];
        let base = (memory.as_ptr() as usize).next_multiple_of(PAGE);
        let (buffer, data) = host()?.shared_buffer(3 * PAGE as u64)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu, base as u64, 3 * PAGE as u64);
        Some((memory, base, shadow))
    }

    /// A write of one byte: the title's into its memory, or a shader's
    /// into the copy.
    fn poke(at: usize, value: u8) {
        // SAFETY: the tests pass addresses inside the memory or the copy.
        unsafe { (at as *mut u8).write_volatile(value) }
    }

    fn peek(at: usize) -> u8 {
        // SAFETY: as in `poke`.
        unsafe { (at as *const u8).read_volatile() }
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn two_shadows_refresh_after_one_aliased_title_write() {
        use victory_loader::linux::{signals, vm};

        let host = host().expect("a Vulkan device is required");
        signals::install();
        let section = vm::Section::new(PAGE as u64).unwrap();
        let addresses: [usize; 2] = std::array::from_fn(|_| {
            vm::map_view(&section, 0, PAGE, 0, vm::PAGE_READWRITE, false).unwrap()
        });
        let shadows = addresses.map(|address| {
            let (buffer, data) = host.shared_buffer(PAGE as u64).unwrap();
            // SAFETY: the buffer is still owned by this shadow.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Shadow::new(buffer, data, gpu, address as u64, PAGE as u64)
        });
        poke(addresses[1] + 19, 37);
        for (address, shadow) in addresses.into_iter().zip(&shadows) {
            assert_eq!(shadow.update(address, PAGE), PAGE);
            assert_eq!(peek(shadow.data as usize + 19), 37);
        }

        // Refreshing the first mapping re-arms the write guard on both.
        poke(addresses[1] + 19, 93);
        let updated = std::array::from_fn::<_, 2, _>(|i| shadows[i].update(addresses[i], PAGE));
        let copied = shadows.each_ref().map(|shadow| peek(shadow.data as usize + 19));
        let clean = std::array::from_fn::<_, 2, _>(|i| shadows[i].update(addresses[i], PAGE));
        for address in addresses {
            vm::unmap_view(address, false).unwrap();
        }
        assert_eq!(updated, [PAGE; 2]);
        assert_eq!(copied, [93; 2]);
        assert_eq!(clean, [0; 2]);
    }

    /// A range declared after ranges on both sides of it, once the snapshot
    /// already spans it: what its refresh brought into the copy is not a
    /// shader's write, and does not go back over what the title wrote since.
    #[test]
    fn a_range_between_earlier_ones_keeps_the_titles_newer_bytes() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let (a, b, c) = (base, base + PAGE, base + 2 * PAGE);
        let copy = |at: usize| s.data as usize + (at - base);
        // An earlier submission read the middle page; the title has written
        // it since.
        s.update(b, PAGE);
        poke(b + 5, 1);
        poke(b + 6, 3);
        // This submission's shaders write the outer pages, then the middle
        // one: each range refreshed, then declared.
        for at in [a, c, b] {
            s.update(at, PAGE);
            s.writing(at, PAGE);
        }
        assert_eq!(peek(copy(b + 5)), 1, "the refresh");
        // While the GPU runs: a shader writes two bytes, the title one.
        poke(copy(a + 7), 9);
        poke(copy(b + 9), 7);
        poke(b + 5, 2);
        for at in [a, c, b] {
            s.copy_back(at, PAGE);
        }
        assert_eq!(peek(b + 5), 2, "the title's newer byte");
        assert_eq!(peek(b + 6), 3, "the title's byte from before");
        // What the shader wrote does go back, in either kind of range.
        assert_eq!((peek(a + 7), peek(b + 9)), (9, 7));
        assert!(s.before.lock().unwrap().is_empty(), "the snapshot is done");
    }

    #[test]
    fn growing_a_sparse_snapshot_only_copies_new_bytes() {
        let Some((_memory, base, s)) = title() else { return };
        s.writing(base + 2 * PAGE, 12);
        let initial = s.before.lock().unwrap()[&(2 * PAGE)].as_ptr();
        s.writing(base, 8);
        s.writing(base + 2 * PAGE + 8, 12);
        s.writing(base, 8);
        {
            let chunks = s.before.lock().unwrap();
            assert_eq!(chunks.values().map(Vec::len).sum::<usize>(), 28);
            assert_eq!(chunks[&(2 * PAGE)].as_ptr(), initial);
        }
        poke(s.data as usize + 2 * PAGE + 13, 3);
        for (offset, size) in [(0, 8), (2 * PAGE, 12), (2 * PAGE + 8, 12)] {
            s.copy_back(base + offset, size);
        }
        assert_eq!(peek(base + 2 * PAGE + 13), 3);
        assert!(s.before.lock().unwrap().is_empty());
    }

    /// Two declared ranges that overlap: the first one's copy-back gives
    /// the copy the title's bytes, which the second must not take for a
    /// shader's and put back over what the title wrote in between.
    #[test]
    fn overlapping_ranges_keep_the_titles_newer_bytes() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let copy = |at: usize| s.data as usize + (at - base);
        s.update(base, 2 * PAGE);
        s.writing(base, PAGE);
        s.writing(base, 2 * PAGE);
        // While the GPU runs: a shader's byte and the title's, in one page.
        poke(copy(base + 3), 9);
        poke(base + 8, 1);
        s.copy_back(base, PAGE);
        assert_eq!((peek(base + 3), peek(base + 8)), (9, 1));
        // The title writes its byte again before the second range's turn.
        poke(base + 8, 2);
        s.copy_back(base, 2 * PAGE);
        assert_eq!(peek(base + 8), 2, "the title's newer byte");
        assert_eq!(peek(base + 3), 9, "the shader's byte");
    }
    /// A title thread writes the page while copy-back has it read: only the
    /// bytes the shader wrote go back, so the title's write stays.
    #[test]
    fn a_title_write_during_copy_back_stays() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let copy = |at: usize| s.data as usize + (at - base);
        s.update(base, PAGE);
        s.writing(base, PAGE);
        poke(copy(base + 100), 9);
        poke(copy(base + 101), 8);
        AFTER_READ.with(|f| *f.borrow_mut() = Some(Box::new(move || poke(base + 4000, 7))));
        s.copy_back(base, PAGE);
        AFTER_READ.with(|f| *f.borrow_mut() = None);
        assert_eq!(
            (peek(base + 100), peek(base + 101)),
            (9, 8),
            "the shader's bytes"
        );
        assert_eq!(peek(base + 4000), 7, "the title's write in between");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn next_draw_sees_a_watched_title_write_during_copy_back() {
        use victory_loader::linux::vm;
        let Some(host) = host() else { return };
        let (buffer, data) = host.shared_buffer(PAGE as u64).unwrap();
        let base = vm::alloc(0, PAGE, vm::MEM_RESERVE | vm::MEM_COMMIT, vm::PAGE_READWRITE, PAGE).unwrap();
        let shadow = Shadow::new(buffer, data, 0, base as u64, PAGE as u64);
        assert_eq!(shadow.update(base, PAGE), PAGE);
        assert_eq!(shadow.update(base, PAGE), 0, "the guest page is guarded");
        shadow.writing(base, PAGE);
        poke(data as usize + 8, 9);
        AFTER_READ.with(|hook| *hook.borrow_mut() = Some(Box::new(move || {
            crate::watch::writing(base + 4000, 1);
            poke(base + 4000, 7);
        })));
        shadow.copy_back(base, PAGE);
        AFTER_READ.with(|hook| *hook.borrow_mut() = None);
        shadow.update(base, PAGE);
        let seen = peek(data as usize + 4000);
        vm::free(base, 0, vm::MEM_RELEASE).unwrap();
        assert_eq!(seen, 7, "the next draw reads the concurrent title update");
    }
}
