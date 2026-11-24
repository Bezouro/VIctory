//! Resolve packed lighting that the title writes as R32_UINT and reads as
//! R9G9B9E5_SHAREDEXP at the same guest address. The two host formats are
//! unrelated DXGI families; a GPU buffer transfers the original texel bits.

use crate::host::{Host, HostList, host};
use crate::object::This;
use crate::resource::HostTexture;
use std::sync::Mutex;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub(crate) struct Layout {
    rows: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    bytes: u64,
}

impl Layout {
    fn new(host: &Host, src: &HostTexture, dst: &HostTexture) -> Option<Self> {
        let (a, b) = unsafe { (src.texture.GetDesc(), dst.texture.GetDesc()) };
        if a.Format != DXGI_FORMAT_R32_UINT
            || b.Format != DXGI_FORMAT_R9G9B9E5_SHAREDEXP
            || a.Dimension != b.Dimension
            || a.Width != b.Width
            || a.Height != b.Height
            || a.DepthOrArraySize != b.DepthOrArraySize
            || a.MipLevels != b.MipLevels
            || a.SampleDesc.Count != 1
            || b.SampleDesc.Count != 1
        {
            return None;
        }
        let layers = if a.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            1
        } else {
            u32::from(a.DepthOrArraySize)
        };
        let count = u32::from(a.MipLevels) * layers;
        if count == 0 {
            return None;
        }
        let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
        let mut bytes = 0;
        // Both formats use exactly four bytes per texel without compression.
        // Only the format changes between the two corresponding footprints.
        unsafe {
            host.device.GetCopyableFootprints(
                &a,
                0,
                count,
                0,
                Some(rows.as_mut_ptr()),
                None,
                None,
                Some(&mut bytes),
            );
        }
        (bytes != 0 && bytes != u64::MAX).then_some(Self { rows, bytes })
    }
}

// Every host list executes on the same queue. All copies through scratch
// are GPU-ordered and lists retain earlier allocations if the slot grows.
fn scratch(host: &Host, bytes: u64) -> Option<ID3D12Resource> {
    static BUFFER: Mutex<Option<(u64, ID3D12Resource)>> = Mutex::new(None);
    let mut slot = BUFFER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((size, buffer)) = slot.as_ref()
        && *size >= bytes
    {
        return Some(buffer.clone());
    }
    let mut buffer = None;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    unsafe {
        host.device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .ok()?;
    }
    let buffer: ID3D12Resource = buffer?;
    *slot = Some((bytes, buffer.clone()));
    Some(buffer)
}

pub(crate) fn copy(list: &HostList, placed: This, address: u64) -> bool {
    let (Some(host), Some(src), Some(dst)) = (
        host(),
        crate::resource::packed_light_source(address, placed),
        crate::resource::host_copy(placed),
    ) else {
        return false;
    };
    let Some(layout) = dst.lighting_layout.get_or_init(|| {
        let layout = Layout::new(host, &src, &dst)?;
        #[cfg(feature = "diagnostics")]
        tracing::info!(
            address = format_args!("{address:#x}"),
            subresources = layout.rows.len(),
            bytes = layout.bytes,
            "packed lighting alias links R32_UINT producer to R9G9B9E5 SRV"
        );
        Some(layout)
    }) else {
        return false;
    };
    let previous = list.recorded_texture_state(&src);
    const READ: u32 = D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32;
    const WRITE: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
    const SHADER: u32 = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;
    if !list.new_packed_light_pair(&dst, &src)
        && previous == Some(READ)
        && list.recorded_texture_state(&dst) == Some(SHADER)
    {
        return true;
    }
    let Some(buffer) = scratch(host, layout.bytes) else {
        return false;
    };
    list.copy_state(&src, READ);
    list.copy_state(&dst, WRITE);
    list.buffer_state(&buffer, WRITE);
    let part = |tex: &ID3D12Resource, i: usize| D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(tex) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: i as u32,
        },
    };
    let row = |r| D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(&buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: r },
    };
    // SAFETY: the source and destination have identical dimensions, layers,
    // mips and texel size. Each copy names its own format and valid row pitch.
    unsafe {
        for (i, r) in layout.rows.iter().enumerate() {
            list.list
                .CopyTextureRegion(&row(*r), 0, 0, 0, &part(&src.texture, i), None);
        }
        list.buffer_state(&buffer, READ);
        for (i, r) in layout.rows.iter().enumerate() {
            let mut r = *r;
            r.Footprint.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
            list.list
                .CopyTextureRegion(&part(&dst.texture, i), 0, 0, 0, &row(r), None);
        }
    }
    // A draw may also write the source through a different binding.
    if let Some(state) = previous
        && state
            & (D3D12_RESOURCE_STATE_RENDER_TARGET.0 | D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0)
                as u32
            != 0
    {
        list.copy_state(&src, state);
    }
    true
}
