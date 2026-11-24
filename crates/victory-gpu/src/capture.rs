//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved
/// too, and their draws traced at debug level.
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list = std::env::var("VICTORY_CAPTURE_FRAMES").unwrap_or_default();
        list.split(',')
            .filter_map(|f| f.trim().parse().ok())
            .collect()
    })
    .contains(&n)
        || n == NEXT_BUSY.load(std::sync::atomic::Ordering::Relaxed)
}

/// `VICTORY_CAPTURE_BUSY=<draws>[,<count>]`: the frame after one that ran
/// more than `<draws>` draws and dispatches on the host is listed too (as if
/// named in `VICTORY_CAPTURE_FRAMES`), up to `<count>` times (default 2),
/// every 8th frame: the world, whose frame numbers vary run to run.
static NEXT_BUSY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Keep the first submission binding a selected pipeline as a standalone RenderDoc
/// capture. Command lists recorded before capture are included when submitted.
#[cfg(target_os = "linux")]
pub(crate) struct SubmissionCapture(extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> u32);

#[cfg(target_os = "linux")]
impl SubmissionCapture {
    pub(crate) fn begin(matches: impl FnOnce(&str) -> bool) -> Option<Self> {
        use std::sync::{OnceLock, atomic::{AtomicBool, Ordering}};
        static FILTER: OnceLock<Option<String>> = OnceLock::new();
        static TAKEN: AtomicBool = AtomicBool::new(false);
        let needle = FILTER.get_or_init(|| std::env::var("VICTORY_CAPTURE_PIPELINE").ok().filter(|s| !s.is_empty())).as_deref()?;
        if TAKEN.load(Ordering::Relaxed) || !matches(needle) || TAKEN.swap(true, Ordering::Relaxed) {
            return None;
        }
        // RenderDoc's public API 1.0.0 prefix, through EndFrameCapture.
        #[repr(C)]
        struct Api {
            preceding: [extern "C" fn(); 19],
            start: extern "C" fn(*mut libc::c_void, *mut libc::c_void),
            capturing: extern "C" fn() -> u32,
            end: extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> u32,
        }
        // SAFETY: look up only the preloaded RenderDoc library's public C entry point.
        let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"RENDERDOC_GetAPI".as_ptr()) };
        if symbol.is_null() {
            tracing::warn!("capture pipeline matched without RenderDoc loaded");
            return None;
        }
        type GetApi = unsafe extern "C" fn(i32, *mut *const Api) -> i32;
        // SAFETY: exact signature of RENDERDOC_GetAPI on Linux.
        let get: GetApi = unsafe { std::mem::transmute(symbol) };
        let mut api = std::ptr::null();
        // SAFETY: RenderDoc owns the versioned table for the lifetime of the process.
        if unsafe { get(10000, &mut api) } != 1 {
            return None;
        }
        let api = unsafe { api.as_ref() }?;
        if (api.capturing)() != 0 {
            return None;
        }
        tracing::info!(needle, "capturing host submission");
        (api.start)(std::ptr::null_mut(), std::ptr::null_mut());
        ((api.capturing)() != 0).then(|| Self(api.end))
    }
}

#[cfg(target_os = "linux")]
impl Drop for SubmissionCapture {
    fn drop(&mut self) {
        let saved = (self.0)(std::ptr::null_mut(), std::ptr::null_mut());
        tracing::info!(saved, "finished host submission capture");
    }
}

/// The first workload-selected frame has been requested for diagnostics.
pub fn busy_capture_started() -> bool {
    NEXT_BUSY.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// At the present of `frame`, after `ran` host draws and dispatches since
/// the last present: lists a later frame if `VICTORY_CAPTURE_BUSY` says so.
pub(crate) fn note_busy(frame: u64, ran: u64) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static SETTING: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    static DONE: AtomicU64 = AtomicU64::new(0);
    static LAST: AtomicU64 = AtomicU64::new(0);
    let Some((draws, count)) = *SETTING.get_or_init(|| {
        let v = std::env::var("VICTORY_CAPTURE_BUSY").ok()?;
        let mut it = v.split(',').map(|x| x.trim().parse::<u64>().ok());
        Some((it.next()??, it.next().flatten().unwrap_or(2)))
    }) else {
        return;
    };
    if ran > draws
        && DONE.load(Relaxed) < count
        && frame >= LAST.load(Relaxed) + 8
        && NEXT_BUSY.load(Relaxed) <= frame
    {
        NEXT_BUSY.store(frame + 1, Relaxed);
        LAST.store(frame + 1, Relaxed);
        DONE.fetch_add(1, Relaxed);
        tracing::info!(frame = frame + 1, ran, "VICTORY_CAPTURE_BUSY: capturing the next frame");
    }
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    folder().is_some() && (n.is_power_of_two() || listed(n) || n == SCENE_CAPTURE.load(std::sync::atomic::Ordering::Relaxed))
}

/// Saves 8-bit RGBA or BGRA rows (`pitch` bytes apart) as frame `n`.
pub(crate) fn save(n: u64, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_as(&format!("frame-{n}"), width, height, pitch, bgra, rows);
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3),
    );
    chunk(&mut png, b"IEND", &[]);
    let path = folder.join(format!("{name}.png"));
    match std::fs::write(&path, png) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

/// Clip input, clip output and pixel shader invocations of selected mesh
/// draws. The host can expose these even without mesh-specific statistics.
pub(crate) struct MeshStats {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    labels: std::sync::Mutex<Vec<(String, MeshCommand)>>,
}

#[derive(Debug)]
pub(crate) enum MeshCommand {
    Compute { groups: Option<[u32; 3]> },
    Direct { groups: u32 },
    Indirect { address: u64, max: u32, counted: bool, sampled: bool },
}

impl MeshStats {
    const LIMIT: u32 = 32;

    const STATS_BYTES: u64 = size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>() as u64;
    const ARG_BYTES: u64 = 32;

    fn storage(list: &crate::host::HostList) -> Option<&Self> {
        list.mesh_stats.get_or_init(|| {
            let (heap, back) = crate::host::query_storage(
                D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, Self::LIMIT,
                Self::STATS_BYTES + Self::ARG_BYTES,
            )?;
            Some(Self { heap, back, labels: Default::default() })
        }).as_ref()
    }

    fn arg_offset(index: u32) -> u64 {
        u64::from(Self::LIMIT) * Self::STATS_BYTES + u64::from(index) * Self::ARG_BYTES
    }

    /// Record the first indirect argument on either side of the host clamp.
    /// The list records one draw at a time. If binding fails, no query is
    /// issued and the next draw can overwrite this unused slot.
    pub(crate) fn args_before(list: &crate::host::HostList, pipeline: crate::object::This,
        args: &ID3D12Resource, offset: u64) -> Option<u32> {
        crate::pipeline::measured_mesh(pipeline)?;
        let stats = Self::storage(list)?;
        let index = stats.labels.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
        if index >= Self::LIMIT { return None; }
        list.buffer_state(args, crate::host::BUFFER_READ);
        // SAFETY: the signature covers a full three-dword mesh record.
        unsafe { list.list.CopyBufferRegion(&stats.back, Self::arg_offset(index), args, offset, 12) };
        Some(index)
    }

    pub(crate) fn args_after(list: &crate::host::HostList, index: u32,
        args: &ID3D12Resource, offset: u64, count: Option<(&ID3D12Resource, u64)>) {
        let stats = list.mesh_stats.get().and_then(Option::as_ref).expect("reserved mesh sample");
        let at = Self::arg_offset(index);
        list.buffer_state(args, crate::host::BUFFER_READ);
        // SAFETY: a matching reservation and the live record held by the list.
        unsafe { list.list.CopyBufferRegion(&stats.back, at + 12, args, offset, 12) };
        if let Some((counter, offset)) = count {
            list.buffer_state(counter, crate::host::BUFFER_READ);
            // SAFETY: the optional execution count covers one dword.
            unsafe { list.list.CopyBufferRegion(&stats.back, at + 24, counter, offset, 4) };
        }
    }

    pub(crate) fn begin(list: &crate::host::HostList, pipeline: crate::object::This,
        command: MeshCommand, targets: &[usize]) -> Option<u32> {
        let label = crate::pipeline::measured_mesh(pipeline)?;
        Self::begin_label(list, label, command, targets)
    }

    pub(crate) fn begin_compute(list: &crate::host::HostList, pipeline: &str, groups: Option<[u32; 3]>) -> Option<u32> {
        if !crate::pipeline::trace_compute(pipeline) {
            return None;
        }
        Self::begin_label(list, pipeline.to_owned(), MeshCommand::Compute { groups }, &[])
    }

    fn begin_label(list: &crate::host::HostList, mut label: String, command: MeshCommand, targets: &[usize]) -> Option<u32> {
        let s = Self::storage(list)?;
        let mut labels = s.labels.lock().unwrap_or_else(|e| e.into_inner());
        let index = labels.len() as u32;
        if index == Self::LIMIT { return None; }
        for &target in targets {
            if let Some(t) = crate::resource::host_resource(target as crate::object::This) {
                use std::fmt::Write;
                // SAFETY: a render target bound on this command list.
                let desc = unsafe { t.GetDesc() };
                let _ = write!(label, " rt={target:#x}/{}x{}/f{}", desc.Width, desc.Height, desc.Format.0);
            }
        }
        labels.push((label, command));
        // SAFETY: an unused query slot on the open command list.
        unsafe { list.list.BeginQuery(&s.heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, index) };
        Some(index)
    }

    pub(crate) fn end(list: &crate::host::HostList, index: Option<u32>) {
        if let (Some(index), Some(Some(s))) = (index, list.mesh_stats.get()) {
            // SAFETY: paired with begin on the same list around a single draw.
            unsafe { list.list.EndQuery(&s.heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, index) };
        }
    }

    pub(crate) fn reset(&self) {
        self.labels.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    pub(crate) fn resolve(&self, list: &ID3D12GraphicsCommandList) {
        let count = self.labels.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
        if count != 0 {
            // SAFETY: each query ended, and the buffer has room for LIMIT.
            unsafe { list.ResolveQueryData(&self.heap, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, count, &self.back, 0) };
        }
    }

    pub(crate) fn report(&self) {
        let labels = self.labels.lock().unwrap_or_else(|e| e.into_inner());
        if labels.is_empty() { return; }
        let mut data = std::ptr::null_mut();
        // SAFETY: the submission fence completed. The mapping is large
        // enough for every resolved entry and remains live while we read.
        unsafe {
            if self.back.Map(0, None, Some(&mut data)).is_err() { return; }
            for (index, ((label, command), stats)) in labels.iter().zip(std::slice::from_raw_parts(
                data.cast::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>(), labels.len(),
            )).enumerate() {
                match *command {
                    MeshCommand::Compute { groups } => {
                        tracing::info!(frame = crate::queue::presented() + 1, pipeline = %label,
                            ?groups, invocations = stats.CSInvocations, "submitted compute statistics");
                        continue;
                    }
                    MeshCommand::Indirect { address, max, counted, sampled: true } => {
                        let words = data.cast::<u8>().add(Self::arg_offset(index as u32) as usize).cast::<u32>();
                        let raw = std::slice::from_raw_parts(words, 3);
                        let clamped = std::slice::from_raw_parts(words.add(3), 3);
                        let count = counted.then(|| *words.add(6));
                        tracing::info!(frame = crate::queue::presented() + 1, pipeline = %label,
                            address = format_args!("{address:#x}"), max, ?raw, ?clamped, ?count, "mesh first indirect record");
                    }
                    MeshCommand::Direct { groups } => {
                        tracing::info!(frame = crate::queue::presented() + 1, pipeline = %label, groups, "mesh direct dispatch");
                    }
                    _ => {}
                }
                tracing::info!(frame = crate::queue::presented() + 1, pipeline = %label,
                    clip_in = stats.CInvocations, clip_out = stats.CPrimitives,
                    pixel_invocations = stats.PSInvocations, "submitted mesh statistics");
            }
            self.back.Unmap(0, None);
        }
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
static TARGETS: std::sync::Mutex<Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// `VICTORY_SHOW_TARGET=<n>`: every frame shows the `n`th texture it
/// rendered to or wrote (first-use order, the `<i>` of a target dump's file
/// names) instead of what the title presents; frames with fewer targets show
/// the title's own picture. `VICTORY_SHOW_TARGET=busiest`: the displayable
/// target (8-bit, 10-bit or 16-bit float colour, 16:9, at least 1280 wide) bound
/// most often (render-target binds and UAV uses, not draws), usually a G-buffer layer
/// in a world frame. Only textures the swap chain copy can take (one mip level,
/// one array slice) are shown, in either mode. A debugging
/// view: it shows a buffer as it is (a G-buffer layer unlit, say), not a
/// finished picture.
#[derive(Clone, Copy)]
pub(crate) enum Show {
    Index(usize),
    Busiest,
}

pub(crate) fn show_index() -> Option<Show> {
    static N: std::sync::OnceLock<Option<Show>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let v = std::env::var("VICTORY_SHOW_TARGET").ok()?;
        let v = v.trim();
        let show = if v.eq_ignore_ascii_case("busiest") {
            Show::Busiest
        } else {
            Show::Index(v.parse().ok()?)
        };
        tracing::warn!(target = v, "VICTORY_SHOW_TARGET: showing a render target instead of the title's picture");
        Some(show)
    })
}

/// Binds (render-target binds and UAV uses) of each noted target this frame
/// (show mode `busiest`); not draws.
static HITS: std::sync::Mutex<Vec<(usize, u32)>> = std::sync::Mutex::new(Vec::new());

/// At the present of `frame` in show mode: the noted target to show, if the
/// frame has one. Clears the frame's list unless a dump still needs it.
pub(crate) fn shown(frame: u64) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    let show = show_index()?;
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    let mut hits = HITS.lock().unwrap_or_else(|e| e.into_inner());
    if frame.is_power_of_two() && frame >= 256 {
        let list: Vec<String> = targets
            .iter()
            .enumerate()
            .map(|(i, (r, c))| {
                // SAFETY: a live resource.
                let d = unsafe { c.texture.GetDesc() };
                let n = hits.iter().find(|h| h.0 == *r).map_or(0, |h| h.1);
                format!("{i}: {}x{} f{} ({n} binds)", d.Width, d.Height, d.Format.0)
            })
            .collect();
        tracing::info!(frame, ?list, "VICTORY_SHOW_TARGET: this frame's targets");
    }
    let pick = match show {
        Show::Index(n) => targets.get(n).filter(|(_, c)| presentable(c)).map(|(_, c)| c.clone()),
        Show::Busiest => targets
            .iter()
            .filter(|(_, c)| {
                // SAFETY: a live resource.
                let d = unsafe { c.texture.GetDesc() };
                presentable(c)
                    && d.Width >= 1280
                    // Screen-shaped (16:9): not a shadow map, which is bound more often.
                    && d.Width * 9 == u64::from(d.Height) * 16
                    && matches!(d.Format.0, 9..=14 | 23..=25 | 27..=32 | 87 | 90 | 91)
            })
            .max_by_key(|(r, _)| hits.iter().find(|h| h.0 == *r).map_or(0, |h| h.1))
            .map(|(_, c)| c.clone()),
    };
    if !(dump_on() && listed(frame)) {
        targets.clear();
    }
    hits.clear();
    pick
}

/// Whether the swap chain copy can take `c` (`CopyResource` needs one mip
/// level and one array slice, like the back buffer's).
fn presentable(c: &crate::resource::HostTexture) -> bool {
    // SAFETY: a live resource.
    let d = unsafe { c.texture.GetDesc() };
    d.MipLevels == 1 && d.DepthOrArraySize == 1
}

/// Optional internal scene submitted as the first presentation plane. Keep
/// the latest matching bind, because the order of targets varies between frames.
static SCENE: std::sync::Mutex<Option<std::sync::Arc<crate::resource::HostTexture>>> =
    std::sync::Mutex::new(None);

fn scene_extent() -> Option<(u64, u32)> {
    static SIZE: std::sync::OnceLock<Option<(u64, u32)>> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let value = std::env::var("VICTORY_SCENE_EXTENT").ok()?;
        let (w, h) = value.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    })
}

/// Capture exponentially spaced frames of the selected scene, even when
/// the first world draws are too sparse to trigger a busy-frame capture.
static SCENE_CAPTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn take_scene(frame: u64) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static COUNT: AtomicU64 = AtomicU64::new(0);
    scene_extent()?;
    let scene = SCENE.lock().unwrap_or_else(|e| e.into_inner()).take()?;
    if folder().is_some() && (COUNT.fetch_add(1, Relaxed) + 1).is_power_of_two() {
        SCENE_CAPTURE.store(frame, Relaxed);
    }
    Some(scene)
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note_target(resource: crate::object::This) {
    if resource.is_null() {
        return;
    }
    if let Some((width, height)) = scene_extent()
        && let Some(copy) = crate::resource::scene_surface(resource, width, height)
    {
        *SCENE.lock().unwrap_or_else(|e| e.into_inner()) = Some(copy);
    }
    let dumping = dump_on() && listed(crate::queue::presented() + 1);
    if !dumping && show_index().is_none() {
        return;
    }
    if show_index().is_some() {
        let mut hits = HITS.lock().unwrap_or_else(|e| e.into_inner());
        match hits.iter_mut().find(|h| h.0 == resource as usize) {
            Some(h) => h.1 += 1,
            None => hits.push((resource as usize, 1)),
        }
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|&(r, _)| r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy));
    }
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        dump_one(host, texture, state, frame, i, *resource as crate::object::This, &d);
    }
}

fn dump_one(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) {
    let Ok(list) = host.new_list() else { return };
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back(&host.device, &list.list, texture);
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        read
    };
    list.close();
    host.execute(&[&list]);
    let Some((buffer, footprint)) = read else { return };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
        if let Some(rgba) = to_rgba(f.Format.0 as u32, f.Width as usize, f.Height as usize, pitch, rows) {
            let name = format!(
                "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, &rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        buffer.Unmap(0, None);
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Rows of a host format as 8-bit RGBA: 8-bit color as is, everything else
/// as channel values scaled so the largest finite one is white (single
/// channels: lowest black, highest white, as grey). `None` for formats not
/// handled.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        87..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=11 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    tracing::info!(format, low, high, "target values");
    for v in values {
        let c = v.map(|c| (((c - low) / range).clamp(0.0, 1.0) * 255.0) as u8);
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }
}
