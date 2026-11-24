//! Single-sample colour texture to title buffer. Record into the destination's
//! normal GPU shadow, so later GPU consumers see the copy in command order.
//! Only the copied texels belong to the GPU; the title owns surrounding bytes.
use super::{COPY_DEST, COPY_SOURCE, CopyLocation, buffer_range, ensure_state};
use crate::{
    host::{HostList, host},
    object::{self, This},
    resource::{self, PlacedFootprint},
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// A recorded title buffer holds its backing memory until the last replay.
pub(crate) struct ReadbackDestination {
    resource: usize,
    #[cfg(feature = "diagnostics")]
    sample: Option<Sample>,
}

#[cfg(feature = "diagnostics")]
struct Sample {
    address: u64,
    format: u32,
    dimensions: [u32; 3],
    row_bytes: usize,
    pitch: u64,
}

#[cfg(feature = "diagnostics")]
impl ReadbackDestination {
    pub(crate) fn report(&self, fence: u64) {
        let Some(s) = &self.sample else { return };
        let mut first = [0u8; 16];
        let bytes = &mut first[..s.row_bytes.min(16)];
        let mapped = crate::host::read_into(s.address as *const u8, bytes);
        tracing::info!(fence, frame=crate::queue::presented()+1, format=s.format,
            dimensions=?s.dimensions, pitch=s.pitch, address=format_args!("{:#x}", s.address),
            first_bytes=mapped.then_some(&*bytes), "texture readback published");
    }
}

impl Drop for ReadbackDestination {
    fn drop(&mut self) {
        resource::resource_release(self.resource as This);
    }
}

pub(super) fn record(
    list: &HostList,
    dst: CopyLocation,
    [x, y, z]: [u32; 3],
    src: CopyLocation,
    region: Option<[u32; 6]>,
) -> Result<(), &'static str> {
    if src.kind != 0 || dst.kind != 1 {
        return Err("requires a texture subresource and a buffer footprint");
    }
    let host = host().ok_or("no host")?;
    let texture = resource::host_resource(src.resource).ok_or("no source texture")?;
    let d = unsafe { texture.GetDesc() };
    let f = dst.footprint;
    let format = resource::pc_format(f.format);
    let own = d.Format.0 as u32;
    let (bytes, block) = resource::element(format).ok_or("unknown texel size")?;
    // Other planes, widened packed targets and subsampled/block formats need a
    // different footprint or a conversion before their bytes reach the title.
    if format != f.format
        || resource::family(format) != resource::family(own)
        || block != 1
        || matches!(format, 19..=22 | 44..=47 | 68..=69)
        || d.SampleDesc.Count != 1
        || d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        return Err("requires a single-sample, single-plane color format with the same storage");
    }
    let sub = src.footprint.offset as u32; // the union's subresource index
    let mips = u32::from(d.MipLevels);
    let volume = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    let layers = if volume {
        1
    } else {
        u32::from(d.DepthOrArraySize)
    };
    if sub >= mips * layers {
        return Err("source subresource out of bounds");
    }
    let mip = sub % mips;
    let w = u32::try_from((d.Width >> mip).max(1)).map_err(|_| "source too wide")?;
    let h = (d.Height >> mip).max(1);
    let depth = if volume {
        (u32::from(d.DepthOrArraySize) >> mip).max(1)
    } else {
        1
    };
    let [l, t, front, r, b, back] = region.unwrap_or([0, 0, 0, w, h, depth]);
    if l >= r || t >= b || front >= back || r > w || b > h || back > depth {
        return Err("source region empty or out of bounds");
    }
    let (width, height, slices) = (r - l, b - t, back - front);
    if x.checked_add(width).is_none_or(|n| n > f.width)
        || y.checked_add(height).is_none_or(|n| n > f.height)
        || z.checked_add(slices).is_none_or(|n| n > f.depth)
        || u64::from(f.row_pitch) < u64::from(f.width) * bytes
    {
        return Err("destination region or pitch out of bounds");
    }
    let row = u64::from(width) * bytes;
    let pitch = u64::from(f.row_pitch);
    let slice = pitch * u64::from(f.height);
    let row_at = |z: u32, y: u32| {
        u64::from(z)
            .checked_mul(slice)?
            .checked_add(u64::from(y) * pitch)
    };
    let offset = row_at(z, y)
        .and_then(|n| n.checked_add(u64::from(x) * bytes))
        .and_then(|n| n.checked_add(f.offset))
        .ok_or("destination offset overflow")?;
    let span = row_at(slices - 1, height - 1)
        .and_then(|n| n.checked_add(row))
        .ok_or("destination size overflow")?;
    let address =
        buffer_range(dst.resource, offset, span).ok_or("destination outside buffer")? as u64;
    let shadow =
        crate::shadow::find_registered(address, span).ok_or("no destination buffer shadow")?;
    let shadow_offset = shadow.offset(address);
    let aligned = shadow_offset.is_multiple_of(512)
        && pitch.is_multiple_of(256)
        && (slices == 1 || height == f.height);
    let host_pitch = if aligned {
        pitch
    } else {
        row.next_multiple_of(256)
    };
    let footprint = PlacedFootprint {
        offset: if aligned { shadow_offset } else { 0 },
        format,
        width,
        height,
        depth: slices,
        row_pitch: u32::try_from(host_pitch).map_err(|_| "host pitch overflow")?,
    };
    let lines = u64::from(height) * u64::from(slices);
    let scratch = if aligned {
        None
    } else {
        let size = (lines - 1)
            .checked_mul(host_pitch)
            .and_then(|n| n.checked_add(row))
            .ok_or("host size overflow")?;
        Some(
            host.texture_copy_scratch(size)
                .ok_or("no texture copy scratch")?,
        )
    };
    let target = scratch.as_ref().unwrap_or(&shadow.buffer);
    if crate::texture::title_written(texture) {
        crate::texture::upload(list, src.resource, texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    list.buffer_state(target, COPY_DEST);
    let mut from = src;
    from.resource = texture.as_raw().cast();
    let to = CopyLocation {
        resource: target.as_raw().cast(),
        kind: 1,
        footprint,
    };
    let source_box = D3D12_BOX {
        left: l,
        top: t,
        front,
        right: r,
        bottom: b,
        back,
    };
    // SAFETY: the two resources are retained by the list's state trackers,
    // bounds and row pitch above cover the box in both locations.
    unsafe {
        list.list.CopyTextureRegion(
            (&raw const to).cast(),
            0,
            0,
            0,
            (&raw const from).cast(),
            Some(&source_box),
        );
    }
    if let Some(scratch) = scratch {
        list.buffer_state(&scratch, COPY_SOURCE);
        list.buffer_state(&shadow.buffer, COPY_DEST);
        for plane in 0..u64::from(slices) {
            for line in 0..u64::from(height) {
                let dst_at = shadow_offset + plane * slice + line * pitch;
                let src_at = (plane * u64::from(height) + line) * host_pitch;
                unsafe {
                    list.list
                        .CopyBufferRegion(&shadow.buffer, dst_at, &scratch, src_at, row);
                }
            }
        }
    }
    // One declared range per row, or one for a contiguous copy. Padding
    // must not even enter the merge when the title writes adjacent bytes.
    let declare = |start, len| {
        shadow.refresh(list, start, len);
        shadow.gpu_writes(list, start, len);
    };
    if row.checked_mul(lines) == Some(span) {
        declare(address, span);
    } else {
        for plane in 0..u64::from(slices) {
            for line in 0..u64::from(height) {
                declare(address + plane * slice + line * pitch, row);
            }
        }
    }
    object::add_ref(dst.resource);
    list.keep_readback(ReadbackDestination {
        resource: dst.resource as usize,
        #[cfg(feature = "diagnostics")]
        sample: {
            static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            TRACE
                .get_or_init(|| {
                    std::env::var_os("VICTORY_TRACE_TEXTURE_READBACK").is_some_and(|v| v == "1")
                })
                .then_some(Sample {
                    address,
                    format,
                    dimensions: [width, height, slices],
                    row_bytes: row as usize,
                    pitch,
                })
        },
    });
    tracing::debug!(
        format,
        sub,
        width,
        height,
        slices,
        guest_pitch = pitch,
        offset,
        aligned,
        "texture readback recorded"
    );
    Ok(())
}
