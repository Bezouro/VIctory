use super::*;

#[test]
fn packed_lighting_shader_samples_both_layers_and_mips_after_gpu_writes() {
    if !super::placed_fixture_process() {
        return;
    }
    use victory_shader::{ResourceType, RootType, UsedResource};

    let Some(host) = host() else {
        eprintln!("skipping packed lighting test without a host GPU");
        return;
    };
    let heap = HostHeap::new(false, 4).unwrap();
    let guest = vec![0u64; 4096];
    let address = guest.as_ptr() as u64;
    let mut src = placed_tex_desc(4, 4, 42, 4);
    src[7] = 2 | (2 << 16);
    let mut dst = src;
    dst[8] = 67;
    dst[12] = 0;
    let placed = crate::resource::texture_at(address, &dst).unwrap();
    assert!(crate::resource::packed_light_source(address, placed).is_none());
    let producer = crate::resource::texture_at(address, &src).unwrap();
    let mut mismatch = dst;
    mismatch[7] = 1 | (2 << 16);
    assert!(
        crate::resource::packed_light_source(
            address,
            crate::resource::texture_at(address, &mismatch).unwrap()
        )
        .is_none()
    );
    assert!(crate::resource::packed_light_source(address + 4, placed).is_none());
    for mip in 0..2 {
        let mut words = placed_tex_uav_desc(address, 42);
        words[1] = 5; // TEXTURE2DARRAY
        words[2] = mip;
        words[4] = 2;
        make(&View::Placed(true, src, words), heap.cpu(mip)).unwrap();
    }
    let mut words = placed_tex_srv_desc(address, 67);
    words[1] = 5;
    words[5] = 2;
    words[7] = 2;
    let view = View::Placed(false, dst, words);
    make(&view, heap.cpu(2)).unwrap();

    let (out, mapped) = host.shared_buffer(64).unwrap();
    let out_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 4,
                StructureByteStride: 16,
                ..Default::default()
            },
        },
    };
    unsafe {
        host.device.CreateUnorderedAccessView(
            &out,
            None::<&ID3D12Resource>,
            Some(&out_desc),
            heap.cpu(3),
        );
    }
    let (write_root, write_pso) = compile_compute(
        r#"
        RWTexture2DArray<uint> m0 : register(u0);
        RWTexture2DArray<uint> m1 : register(u1);
        [RootSignature("DescriptorTable(UAV(u0,numDescriptors=2))")]
        [numthreads(4,4,1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint rgb = (16u<<27) | ((32u + id.y)<<18) | ((64u + id.x)<<9);
            m0[id] = rgb | (256u + id.z);
            if (all(id.xy < 2u)) m1[id] = rgb | (384u + id.z);
        }"#,
        &host.device,
    );
    let (read_root, read_pso) = compile_compute(
        r#"
        Texture2DArray<float4> packed : register(t0);
        RWStructuredBuffer<float4> result : register(u0);
        [RootSignature("DescriptorTable(SRV(t0)),DescriptorTable(UAV(u0))")]
        [numthreads(2,2,1)]
        void main(uint3 id : SV_DispatchThreadID) {
            result[id.y*2 + id.x] = packed.Load(int4(1,1,id.y,id.x));
        }"#,
        &host.device,
    );

    let write = host.new_list().unwrap();
    let read = host.new_list().unwrap();
    let usage = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 0,
    };
    // Record the consumer first, submit it second. No bytes return to
    // guest RAM between the GPU producer and the shader's placed SRV.
    crate::draw::readable(&read, &usage, &view);
    read.buffer_state(&out, crate::host::BUFFER_WRITE);
    unsafe {
        read.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        read.list.SetComputeRootSignature(&read_root);
        read.list.SetPipelineState(&read_pso);
        read.list.SetComputeRootDescriptorTable(0, heap.gpu(2));
        read.list.SetComputeRootDescriptorTable(1, heap.gpu(3));
        read.list.Dispatch(1, 1, 1);
    }
    read.close();
    crate::list::ensure_state(&write, producer, 8);
    unsafe {
        write.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        write.list.SetComputeRootSignature(&write_root);
        write.list.SetPipelineState(&write_pso);
        write.list.SetComputeRootDescriptorTable(0, heap.gpu(0));
        write.list.Dispatch(1, 1, 2);
    }
    write.close();
    host.execute(&[&write, &read]);
    let actual = unsafe { std::slice::from_raw_parts(mapped.cast::<[f32; 4]>(), 4) };
    for layer in 0..2 {
        for (mip, red) in [256, 384].into_iter().enumerate() {
            assert_eq!(
                actual[layer * 2 + mip],
                [
                    (red + layer) as f32 / 256.0,
                    65.0 / 256.0,
                    33.0 / 256.0,
                    1.0
                ]
            );
        }
    }
    assert!(guest.iter().all(|&v| v == 0));
}
