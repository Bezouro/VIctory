//! Single-plane depth images share their format family with an unordered texture.
//! A D32+stencil image instead copies its two aspects through the pitches supplied
//! by the device. Its DSV and shader views continue to refer to the depth resource.
use super::HostTexture;
use crate::host::{HostList, host};
use std::sync::Arc;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(crate) enum DepthTwin {
    Whole(Arc<HostTexture>),
    Planes {
        planes: [Arc<HostTexture>; 2],
        scratch: ID3D12Resource,
        footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    },
}

fn tracked(texture: ID3D12Resource) -> Arc<HostTexture> {
    Arc::new(HostTexture {
        lighting_layout: Default::default(),
        texture,
        state: 0.into(),
        twin: None,
    })
}

fn allocate(desc: &D3D12_RESOURCE_DESC) -> Option<ID3D12Resource> {
    let device = &host()?.device;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: a complete resource description and a live output.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .inspect_err(|e| tracing::warn!(error=%e, ?desc,"depth/stencil twin allocation failed"))
    .ok()?;
    resource
}

impl DepthTwin {
    pub(crate) fn new(depth: &ID3D12Resource) -> Option<Self> {
        // SAFETY: a live depth texture.
        let d = unsafe { depth.GetDesc() };
        if super::family(d.Format.0 as u32) != 19
            || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || d.SampleDesc.Count != 1
        {
            return super::host_twin(depth).map(|r| Self::Whole(tracked(r)));
        }
        let formats = [DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R8_TYPELESS];
        let subs = usize::from(d.MipLevels) * usize::from(d.DepthOrArraySize);
        let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2 * subs];
        let mut size = 0;
        // SAFETY: every requested plane/mip/layer has an output slot.
        unsafe {
            host()?.device.GetCopyableFootprints(
                &d,
                0,
                footprints.len() as u32,
                0,
                Some(footprints.as_mut_ptr()),
                None,
                None,
                Some(&mut size),
            )
        };
        if size == 0
            || size == u64::MAX
            || footprints
                .iter()
                .enumerate()
                .any(|(i, f)| f.Footprint.Format != formats[i / subs])
        {
            tracing::warn!("depth/stencil plane footprints do not match their writable images");
            return None;
        }
        let plane = |format| {
            allocate(&D3D12_RESOURCE_DESC {
                Format: format,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                Alignment: 0,
                ..d
            })
            .map(tracked)
        };
        let planes = [plane(formats[0])?, plane(formats[1])?];
        let scratch = host()?.texture_copy_scratch(size)?;
        tracing::debug!(
            width = d.Width,
            height = d.Height,
            subresources = subs,
            scratch_bytes = size,
            "depth/stencil UAV planes ready"
        );
        Some(Self::Planes {
            planes,
            scratch,
            footprints,
        })
    }

    pub(crate) fn planes(&self) -> &[Arc<HostTexture>] {
        match self {
            Self::Whole(r) => std::slice::from_ref(r),
            Self::Planes { planes, .. } => planes,
        }
    }

    pub(crate) fn copy(&self, list: &HostList, depth: &Arc<HostTexture>, to_depth: bool) {
        let Self::Planes {
            planes,
            scratch,
            footprints,
        } = self
        else {
            let plane = &self.planes()[0];
            if to_depth {
                list.copy_texture(depth, plane);
            } else {
                list.copy_texture(plane, depth);
            }
            return;
        };
        const SOURCE: u32 = 0x800;
        const DEST: u32 = 0x400;
        list.copy_state(depth, if to_depth { DEST } else { SOURCE });
        for plane in planes {
            list.copy_state(plane, if to_depth { SOURCE } else { DEST });
        }
        let subs = footprints.len() / 2;
        tracing::trace!(to_depth, subresources = subs, "recording depth/stencil plane copies");
        // Using the ordinary list buffer tracker gives scratch a state prelude and
        // restores COMMON on close, including when this list is recorded and replayed.
        for filling in [true, false] {
            list.buffer_state(scratch, if filling { DEST } else { SOURCE });
            for (i, &footprint) in footprints.iter().enumerate() {
                // When restoring depth, the writable planes fill scratch. When
                // preparing a UAV, they consume it after the depth image fills it.
                let plane_image = filling == to_depth;
                let (image, sub) = if plane_image {
                    (&planes[i / subs].texture, i % subs)
                } else {
                    (&depth.texture, i)
                };
                let texture = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: the state tracker retains all resources through execution.
                    pResource: unsafe { std::mem::transmute_copy(image) },
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: sub as u32,
                    },
                };
                let buffer = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: the buffer is retained by the list.
                    pResource: unsafe { std::mem::transmute_copy(scratch) },
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: footprint,
                    },
                };
                let (dest, source) = if filling {
                    (&buffer, &texture)
                } else {
                    (&texture, &buffer)
                };
                // SAFETY: complete plane, footprint returned by the device, both in their copy states.
                unsafe { list.list.CopyTextureRegion(dest, 0, 0, 0, source, None) };
            }
        }
    }
}
