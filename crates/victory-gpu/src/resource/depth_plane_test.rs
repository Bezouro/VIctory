//! Test the plane copies for a writable R32G8X24 depth/stencil image on a D3D12 host.
use super::*;
use crate::descriptor::{View, host::HostHeap};
use crate::host::{host, readback_buffer};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

const SUBS: usize = 4; // mip + layer * 2

fn sub(r: &ID3D12Resource, i: usize) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed; the caller keeps the resource through execution.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: i as u32,
        },
    }
}

fn placed(
    r: &ID3D12Resource,
    f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed; the caller keeps the resource through execution.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    }
}

/// Title calls use the console WIN64 vtable even on Linux.
#[repr(C)]
struct TitleBarrier(u32, u32, This, u32, u32, u32, u32);

fn title_list() -> (This, This) {
    let null = std::ptr::null();
    let (mut allocator, mut list) = (null, null);
    let iid = &PAGEABLE_IIDS[1];
    assert_eq!(
        crate::list::create_allocator(null, 0, iid, &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(null, 0, 0, allocator, null, iid, &mut list),
        S_OK
    );
    (allocator, list)
}
#[test]
#[ignore = "requires a D3D12 host and DXC"]
fn depth_stencil_plane_copies_preserve_compute_writes_and_neighbors() {
    for finish_at_close in [true, false] {
        check(
            19,
            &[DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R8_TYPELESS],
            finish_at_close,
        );
    }
    for format in [39, 40] {
        check(format, &[DXGI_FORMAT_R32_TYPELESS], false);
    }
}

fn check(format: u32, formats: &[DXGI_FORMAT], finish_at_close: bool) {
    let host = host().expect("plane copy test needs D3D12");
    let device = &host.device;
    let declared = ResourceDesc {
        dimension: 3,
        width: 8,
        height: 8,
        depth_or_array_size: 2,
        mip_levels: 2,
        format,
        sample_count: 1,
        sample_quality: 0,
        flags: 6,
        alignment: 0,
        layout: 0,
    };
    let props = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };
    let mut handle = std::ptr::null();
    assert_eq!(
        create_committed(
            std::ptr::null(),
            &props,
            0,
            &declared,
            0x10,
            std::ptr::null(),
            &RESOURCE_IIDS[0],
            &mut handle
        ),
        S_OK
    );
    let depth = host_resource(handle).unwrap();
    let desc = unsafe { depth.GetDesc() };
    let heap = HostHeap::new(false, formats.len() as u32).unwrap();
    for (plane, view_format) in (if format == 19 {
        &[21, 22][..]
    } else {
        &[41][..]
    })
    .iter()
    .enumerate()
    {
        let words = [*view_format, 5, 1, 1, 1, plane as u32, 0, 0, 0, 0];
        let made = crate::descriptor::host::texture_uav_desc(handle, Some(&words))
            .flatten()
            .expect("title UAV descriptor must be materializable");
        assert_eq!(
            made.Format,
            [DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8_UINT][plane]
        );
        assert_eq!(unsafe { made.Anonymous.Texture2DArray.PlaneSlice }, 0);
        heap.make_at(
            plane as u32,
            &View::Unordered(handle as usize, 0, Some(words)),
        );
    }
    if format == 19 {
        for (view, plane) in [(21, 1), (22, 0), (62, 2)] {
            assert!(
                crate::descriptor::host::texture_uav_desc(
                    handle,
                    Some(&[view, 4, 0, plane, 0, 0, 0, 0, 0, 0])
                )
                .is_none()
            );
        }
        let stencil = crate::descriptor::host::texture_uav_desc(
            handle,
            Some(&[62, 4, 0, 1, 0, 0, 0, 0, 0, 0]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(unsafe { stencil.Anonymous.Texture2D.PlaneSlice }, 0);
    }
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); SUBS * formats.len()];
    let mut size = 0;
    // SAFETY: one output slot per subresource.
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            layouts.len() as u32,
            0,
            Some(layouts.as_mut_ptr()),
            None,
            None,
            Some(&mut size),
        )
    };
    for (p, route) in layouts.chunks_exact(SUBS).enumerate() {
        eprintln!(
            "plane={p} footprint={:?} pitch={} offset={} bytes={size}",
            route[0].Footprint.Format, route[0].Footprint.RowPitch, route[0].Offset
        );
        for f in route {
            assert_eq!(
                f.Footprint.Format, formats[p],
                "plane conversion would be required"
            );
        }
    }
    let output = readback_buffer(size).expect("plane readback");
    let source = r#"
        RWTexture2DArray<float> depth : register(u0);
        #ifdef STENCIL
        RWTexture2DArray<uint> stencil : register(u1);
        #endif
        [RootSignature("DescriptorTable(UAV(u0, numDescriptors=2))")]
        [numthreads(4,4,1)] void main(uint3 id : SV_DispatchThreadID) {
            if (id.x == 1) depth[id] = 0.8125;
            #ifdef STENCIL
            if (id.y == 2) stencil[id] = 0x9d;
            #endif
        }"#;
    let source = if format == 19 {
        format!("#define STENCIL\n{source}")
    } else {
        source.replace("numDescriptors=2", "numDescriptors=1")
    };
    let code = victory_shader::hostlib::compile_hlsl(&source, "main", "cs_6_0").unwrap();
    // SAFETY: valid root signature, compiled code and descriptor heap description.
    let (root, pso) = unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).unwrap();
        let pso: ID3D12PipelineState = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .unwrap();
        (root, pso)
    };
    let (allocator, title) = title_list();
    let first = crate::list::host_list(title).unwrap();
    let barriers = crate::object::slot::<extern "win64" fn(This, u32, *const TitleBarrier) -> u32>(
        title, 0xd0,
    );
    crate::list::ensure_state(&first, handle, 0x10);
    let l = &first.list;
    // SAFETY: resources live through both submissions. All copies use complete subresources.
    unsafe {
        for s in 0..SUBS {
            let dsv = host.dsv(
                &depth,
                Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: if format == 19 {
                        DXGI_FORMAT_D32_FLOAT_S8X24_UINT
                    } else {
                        DXGI_FORMAT_D32_FLOAT
                    },
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                    Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                            MipSlice: (s % 2) as u32,
                            FirstArraySlice: (s / 2) as u32,
                            ArraySize: 1,
                        },
                    },
                    ..Default::default()
                }),
            );
            l.ClearDepthStencilView(
                dsv,
                if format == 19 {
                    D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL
                } else {
                    D3D12_CLEAR_FLAG_DEPTH
                },
                (s + 1) as f32 / 8.0,
                0x31 + s as u8,
                None,
            );
        }
        let to_uav = TitleBarrier(0, 0, handle, u32::MAX, 0x10, 8, 0);
        barriers(title, 1, &to_uav);
        l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        l.SetComputeRootSignature(&root);
        l.SetComputeRootDescriptorTable(0, heap.gpu(0));
        l.SetPipelineState(&pso);
        l.Dispatch(1, 1, 1);
        if !finish_at_close {
            let back = [
                TitleBarrier(2, 0, handle, 0, 0, 0, 0),
                TitleBarrier(0, 0, handle, u32::MAX, 8, 0x10, 0),
            ];
            barriers(title, back.len() as u32, back.as_ptr());
            let dsv = host.dsv(
                depth,
                Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: if format == 19 {
                        DXGI_FORMAT_D32_FLOAT_S8X24_UINT
                    } else {
                        DXGI_FORMAT_D32_FLOAT
                    },
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                    Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                            MipSlice: 1,
                            FirstArraySlice: 1,
                            ArraySize: 1,
                        },
                    },
                    ..Default::default()
                }),
            );
            use windows::Win32::Foundation::RECT;
            l.ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAG_DEPTH,
                0.9375,
                0,
                Some(&[RECT {
                    left: 0,
                    top: 0,
                    right: 1,
                    bottom: 1,
                }]),
            );
        }
    }
    assert_eq!(
        crate::object::slot::<extern "win64" fn(This) -> HRESULT>(title, 0x48)(title),
        S_OK
    );
    if finish_at_close {
        host.execute(&[first]);
    }
    let second = host.new_list().unwrap();
    let l = &second.list;
    crate::list::ensure_state(&second, handle, 0x800);
    // SAFETY: complete subresources and footprints queried from the device.
    unsafe {
        for (i, &f) in layouts.iter().enumerate() {
            l.CopyTextureRegion(&placed(&output, f), 0, 0, 0, &sub(&depth, i), None);
        }
    }
    second.close();
    if finish_at_close {
        host.execute(&[&second]);
    } else {
        // Replaying the same closed list must prepare the scratch and both planes again.
        host.execute(&[first, first, &second]);
    }
    let mut mapped = std::ptr::null_mut();
    // SAFETY: the fence is complete. Footprints and pitches are those returned by the device.
    unsafe {
        output
            .Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: size as usize,
                }),
                Some(&mut mapped),
            )
            .unwrap();
        for (i, f) in layouts.iter().enumerate() {
            let (p, s) = (i / SUBS, i % SUBS);
            for y in 0..f.Footprint.Height as usize {
                let row = mapped
                    .cast::<u8>()
                    .add(f.Offset as usize + y * f.Footprint.RowPitch as usize);
                for x in 0..f.Footprint.Width as usize {
                    if p == 0 {
                        let want = if !finish_at_close && s == 3 && x == 0 && y == 0 {
                            0.9375
                        } else if s == 3 && x == 1 {
                            0.8125
                        } else {
                            (s + 1) as f32 / 8.0
                        };
                        assert_eq!(
                            row.cast::<f32>().add(x).read(),
                            want,
                            "depth sub {s} at {x},{y}"
                        );
                    } else {
                        let want = if s == 3 && y == 2 {
                            0x9d
                        } else {
                            0x31 + s as u8
                        };
                        assert_eq!(row.add(x).read(), want, "stencil sub {s} at {x},{y}");
                    }
                }
            }
        }
        output.Unmap(0, Some(&D3D12_RANGE::default()));
    }
    for object in [title, allocator] {
        assert_eq!(
            crate::object::slot::<extern "win64" fn(This) -> u32>(object, 0x10)(object),
            0
        );
    }
    assert_eq!(resource_release(handle), 0);
    eprintln!(
        "title format {format}, copy on close={finish_at_close}: 160 texels matched per plane, {} planes, 2 mips and 2 layers",
        formats.len()
    );
}
