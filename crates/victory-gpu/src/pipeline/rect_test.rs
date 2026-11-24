//! End-to-end synthetic DrawIndirect through the same state, clamp and shader
//! pipeline as VS_TiledLighting. The title identifiers only select its route.
use super::*;
use crate::pipeline::{Shader, Translated};
use std::sync::OnceLock;
use windows::Win32::Foundation::RECT;

const VERTEX: &str = r#"
struct V {float4 p:SV_Position; float u:TEXCOORD0;
          nointerpolation uint flatU:TEXCOORD1; nointerpolation float flatF:TEXCOORD2;};
V main(uint id:SV_VertexID,uint inst:SV_InstanceID) {
    uint tile=id/3, c=id%3;
    V o;
    o.p=float4(-1.0+tile+(c==2),1.0-inst-(c!=0),0.5,1.0);
    o.u=c==2;
    o.flatU=2+tile*12+inst*24+c;
    o.flatF=1+tile*4+inst*8+c;
    return o;
}"#;
const PIXEL: &str = r#"
float4 main(float4 p:SV_Position,float u:TEXCOORD0,
    nointerpolation uint flatU:TEXCOORD1,nointerpolation float flatF:TEXCOORD2):SV_Target {
  return float4(5,round(u*8),flatU,flatF)/255.0;
}"#;

fn synthetic(stage: &'static str, hash: &str, source: &str, target: &str) -> Arc<Shader> {
    let dxil = victory_shader::hostlib::compile_hlsl(source, "main", target).unwrap();
    let translated = Translated {
        host: victory_shader::emit::HostShader {
            stage: if stage == "vertex" {
                victory_shader::HwStage::Vs
            } else {
                victory_shader::HwStage::Ps
            },
            dxil,
            user_data: vec![],
            samplers: vec![],
            static_samplers: vec![],
            root_cbvs: vec![],
            root_srvs: vec![],
            root_uavs: vec![],
            stride_table: false,
            indexed_reads: vec![],
        },
        used: vec![],
    };
    Arc::new(Shader {
        stage,
        hash: hash.into(),
        container: vec![],
        translated: OnceLock::from(Some(translated)),
        #[cfg(feature = "diagnostics")]
        wave64: OnceLock::new(),
        #[cfg(feature = "diagnostics")]
        native_uavs: OnceLock::new(),
    })
}

#[test]
#[ignore = "requires a D3D12 Vulkan host and DXC"]
fn procedural_rectangles_keep_indirect_records_and_flat_attributes() {
    run(false);
    if host().expect("Vulkan host").depth_bounds_supported {
        run(true);
    }
}

fn run(depth_bounds: bool) {
    let host = host().expect("Vulkan host");
    let vs = synthetic("vertex", "25ebc92075b62822", VERTEX, "vs_6_0");
    let ps = synthetic("pixel", "synthetic-tiled-pixel", PIXEL, "ps_6_0");
    let vs_bytes = &vs.translated().unwrap().host.dxil;
    let mut observes_vertices = ps.translated().unwrap().host.dxil.clone();
    let feature = victory_shader::Container::parse(&observes_vertices)
        .unwrap()
        .part(victory_shader::FourCC::SFI0)
        .unwrap()
        .offset
        + 8;
    observes_vertices[feature + 2] |= 2;
    assert!(
        victory_shader::emit::procedural_rectangle(vs_bytes, &observes_vertices)
            .is_err_and(|reason| reason.contains("original triangle vertices"))
    );
    let mut g = GraphicsState::default();
    g.depth.DepthEnable = depth_bounds.into();
    g.depth_bounds = depth_bounds;
    if depth_bounds {
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        g.depth.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        g.depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    }
    g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE(RECTANGLES);
    g.targets.NumRenderTargets = 1;
    g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    let blend = &mut g.blend.RenderTarget[0];
    blend.BlendEnable = true.into();
    blend.DestBlend = D3D12_BLEND_ONE;
    blend.DestBlendAlpha = D3D12_BLEND_ONE;
    let mut pso = std::ptr::null();
    assert_eq!(
        crate::pipeline::hand_out(
            vec![vs.clone(), ps.clone()],
            Some(Box::new(g)),
            &crate::pipeline::PIPELINE_IIDS[0],
            &mut pso
        ),
        0
    );
    let ready = crate::pipeline::host_pipeline(pso).expect("graphics PSO");
    if !ready.rect_gs {
        panic!(
            "GS refused: {:?}",
            victory_shader::emit::procedural_rectangle(
                &vs.translated().unwrap().host.dxil,
                &ps.translated().unwrap().host.dxil
            )
            .err()
        );
    }
    // The uint and float at the provoking vertex differ from those at the
    // shared rectangle corner. Both must remain flat across the diagonal.
    let texture = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 8,
        Height: 8,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    // SAFETY: all D3D descriptions and mapped memory live until the synchronous
    // submission returns. Every row of readback uses the reported row pitch.
    unsafe {
        let mut target: Option<ID3D12Resource> = None;
        host.device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &texture,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut target,
            )
            .unwrap();
        let target = target.unwrap();
        let heap: ID3D12DescriptorHeap = host
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap();
        let rtv = heap.GetCPUDescriptorHandleForHeapStart();
        host.device.CreateRenderTargetView(&target, None, rtv);
        let depth = depth_bounds.then(|| {
            let desc = D3D12_RESOURCE_DESC {
                Format: DXGI_FORMAT_D32_FLOAT,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                ..texture
            };
            let mut surface: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    None,
                    &mut surface,
                )
                .unwrap();
            let surface = surface.unwrap();
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let handle = heap.GetCPUDescriptorHandleForHeapStart();
            host.device.CreateDepthStencilView(&surface, None, handle);
            (surface, heap, handle)
        });
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut size = 0;
        host.device.GetCopyableFootprints(
            &texture,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut size),
        );
        let readback = crate::host::readback_buffer(size).unwrap();

        for count in 0u32..=2 {
            let mem = crate::resource::Memory::zeroed(128, 64).unwrap();
            let address = mem.address();
            crate::shadow::register(address as usize, address, 128);
            let shadow = crate::shadow::find(address, 128).unwrap();
            // Initially zero records and count in guest memory. A prior GPU
            // copy supplies two 32-byte records at +16 and the count at +96.
            let (upload, map) = host.upload_buffer(128).unwrap();
            let words = std::slice::from_raw_parts_mut(map.cast::<u32>(), 32);
            words.fill(0x66666666);
            words[4..8].copy_from_slice(&[3, 1, 11, 3]);
            words[12..16].copy_from_slice(&[6, 2, 17, 9]);
            words[24] = count;
            let list = host.new_list().unwrap();
            list.buffer_state(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
            list.list
                .CopyBufferRegion(&shadow.buffer, 0, &upload, 0, 128);
            let mut state = crate::draw::State::default();
            state.set_pipeline(pso);
            state.set_topology(&list, 17);
            list.list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 8.,
                Height: 8.,
                MaxDepth: 1.,
                ..Default::default()
            }]);
            list.list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 8,
                bottom: 8,
            }]);
            if let Some((_, _, dsv)) = &depth {
                list.list
                    .ClearDepthStencilView(*dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None);
            }
            list.list.OMSetRenderTargets(
                1,
                Some(&rtv),
                false,
                depth.as_ref().map(|d| &d.2 as *const _),
            );
            list.list.ClearRenderTargetView(rtv, &[0.; 4], None);
            assert!(
                !state.execute_indirect(&list, (1, 32, 2), address + 16, Some(address + 96)),
                "indexed rectangles still require an index-aware route"
            );
            assert!(
                state.execute_indirect(&list, (0, 32, 2), address + 16, Some(address + 96)),
                "procedural indirect draw must reach the host"
            );
            list.list.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyTextureRegion(
                &D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: footprint,
                    },
                },
                0,
                0,
                0,
                &D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&target),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                },
                None,
            );
            list.list.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
            list.close();
            host.execute(&[&list]);
            let mut at = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut at)).unwrap();
            let mut pixels = 0;
            for y in 0usize..8 {
                for x in 0usize..8 {
                    let (tx, ty) = ((x / 4) as u8, (y / 4) as u8);
                    let draws = u8::from(count >= 1 && tx == 0 && ty == 0) + u8::from(count == 2);
                    let expected = [
                        5,
                        (x % 4 * 2 + 1) as u8,
                        2 + tx * 12 + ty * 24,
                        1 + tx * 4 + ty * 8,
                    ]
                    .map(|v| v * draws);
                    let got = std::slice::from_raw_parts(
                        at.cast::<u8>()
                            .add(y * footprint.Footprint.RowPitch as usize + x * 4),
                        4,
                    );
                    assert_eq!(got, expected, "count={count} at ({x},{y})");
                    pixels += usize::from(draws > 0);
                }
            }
            readback.Unmap(0, None);
            // Nonzero first vertex/instance offsets and record padding are
            // preserved as well as the data the shader can observe.
            let mut raw = std::ptr::null_mut();
            shadow.buffer.Map(0, None, Some(&mut raw)).unwrap();
            let consumed = std::slice::from_raw_parts(raw.cast::<u32>(), 32);
            assert_eq!(&consumed[4..8], &words[4..8]);
            assert_eq!(&consumed[12..16], &words[12..16]);
            assert_eq!(consumed[8], 0x66666666);
            shadow.buffer.Unmap(0, None);
            crate::shadow::unregister(address as usize, address);
            eprintln!(
                "GPU procedural rectangle depth_bounds={depth_bounds} count={count}: {pixels}/64 covered; 64 RGBA values matched"
            );
        }
    }
    assert_eq!(crate::pipeline::release(pso), 0);
}
