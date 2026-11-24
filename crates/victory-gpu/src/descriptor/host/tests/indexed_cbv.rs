use super::*;
use crate::descriptor::{HEAP_IIDS, HeapDesc, Placement};
use crate::object::This;
use victory_shader::{ResourceType, RootType, UsedResource, resource::IndexedRead};

#[test]
fn cbv_index_materializes_the_srv_seen_by_the_gpu_at_submission() {
    if !super::placed_fixture_process() {
        return;
    }
    let Some(host) = host() else {
        eprintln!("no graphics device");
        return;
    };
    // Keep the guest heap, CBV, and both known buffers alive through the fence.
    let mut h: This = std::ptr::null();
    #[repr(C)]
    struct Description(HeapDesc, Placement);
    let desc = Description(
        HeapDesc {
            kind: 0,
            count: 16,
            flags: 1,
            nodes: 0,
        },
        Placement {
            address: 0,
            size: 0,
        },
    );
    assert_eq!(
        crate::descriptor::create_heap(std::ptr::null(), &desc.0, &HEAP_IIDS[0], &mut h),
        0
    );
    let heap = crate::descriptor::heap(h);
    let mirrored = heap.host().unwrap();
    let mut constants = [0u32; 24];
    let cb = constants.as_mut_ptr() as u64;
    let mut buffers = [[0u32; 72]; 2];
    for (n, words) in buffers.iter_mut().enumerate() {
        for (i, w) in words.iter_mut().enumerate() {
            *w = ((n as u32 + 1) << 24) | i as u32;
        }
        let base = words.as_mut_ptr() as u64;
        crate::shadow::register(base as usize, base, 288);
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 288;
        let mut view = [0u32; 16];
        view[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        view[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        view[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view[4] = 4; // FirstElement, so the view begins at byte 16
        view[6] = 64;
        view[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        view[10] = base as u32;
        view[11] = (base >> 32) as u32;
        crate::descriptor::create_placed_srv(
            std::ptr::null(),
            &resource,
            &view,
            heap.start + 32 * (4 + n as u64),
        );
    }
    crate::descriptor::create_cbv(
        std::ptr::null(),
        &[cb as u32, (cb >> 32) as u32, 96, 0],
        heap.start + 256,
    );
    assert!(
        mirrored.indexed_srv(heap, 8).is_none(),
        "CBV is not a buffer SRV"
    );
    assert!(
        mirrored.indexed_srv(heap, 16).is_none(),
        "beyond the guest heap"
    );
    let (root, pso) = compile_compute(
        r#"
        cbuffer selectBuffer : register(b0) { uint4 cb[6]; }
        ByteAddressBuffer inputs[6] : register(t0);
        RWByteAddressBuffer result : register(u0);
        [RootSignature("DescriptorTable(CBV(b0)),DescriptorTable(SRV(t0,numDescriptors=6)),UAV(u0)")]
        [numthreads(1,1,1)]
        void main() {
            uint i = cb[4].x;
            result.Store4(0, inputs[i].Load4(0x60));
            result.Store4(16, inputs[i].Load4(0x70));
            result.Store2(32, inputs[i].Load2(0x90));
            uint old; result.InterlockedAdd(40, 1, old);
            result.Store(44 + 4*old, inputs[i].Load(0x60));
        }
    "#,
        &host.device,
    );
    let (next_root, next_pso) = compile_compute(
        r#"
        RWByteAddressBuffer result : register(u0);
        [RootSignature("UAV(u0)")] [numthreads(1,1,1)]
        void main() { uint old; result.InterlockedAdd(112, 1, old); }
    "#,
        &host.device,
    );
    let (out, mapped) = host.shared_buffer(128).unwrap();
    unsafe {
        mapped.write_bytes(0, 128);
    }
    let get = |index: usize| unsafe { mapped.cast::<u32>().add(index).read_volatile() };
    let used = UsedResource {
        kind: ResourceType::Cbv,
        root: RootType::Table,
        table_offset_dwords: 64,
        lower_bound: 7,
        stride: 0,
        dwords: 4,
        user_data: 5,
        slot: 1,
    };
    let read = IndexedRead {
        slot: 1,
        table_offset_dwords: Some(64),
        dword: 16,
        byte_offset: 0,
    };
    let record = |conflict: Option<u32>| {
        let mut state = crate::draw::State::default();
        state.set_heaps(&[h]);
        state.set_table(true, 1, heap.start);
        let list = host.new_list().unwrap();
        let slot = state.prepared_cbv_table(&list, &used, read);
        if let Some(wanted) = conflict {
            let base = buffers[1].as_ptr() as u64;
            let shadow = crate::shadow::find_registered(base, 288).unwrap();
            list.buffer_state(&shadow.buffer, wanted);
        }
        list.buffer_state(&out, crate::host::BUFFER_WRITE);
        // SAFETY: open list, initialized signature/pipeline, valid output allocation.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(mirrored.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, mirrored.gpu(slot));
            list.list.SetComputeRootDescriptorTable(1, mirrored.gpu(0));
            list.list
                .SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            state.indexed_command(&list, || list.list.Dispatch(1, 1, 1));
            list.list.SetComputeRootSignature(&next_root);
            list.list.SetPipelineState(&next_pso);
            list.list
                .SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        (state, list)
    };
    constants[16] = 4;
    let (_state, list) = record(None);
    for index in [5u32, 4] {
        constants[16] = index;
        std::hint::black_box(&constants);
        host.execute(&[&list]);
        let result = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), 10) };
        let tag = (index - 3) << 24;
        let want = [28, 29, 30, 31, 32, 33, 34, 35, 40, 41].map(|v| v | tag);
        assert_eq!(result, want, "CBV heap index {index}");
        eprintln!("CBV +0x40 = {index}, GPU headers and vertex = {result:x?}");
    }
    // Slot 5 remains materialized from the first replay; a late selector
    // cannot read it in UAV or COPY_DEST state. Only that dispatch is skipped.
    for (iteration, conflict) in [
        crate::host::BUFFER_WRITE,
        D3D12_RESOURCE_STATE_COPY_DEST.0 as u32,
    ]
    .into_iter()
    .enumerate()
    {
        constants[16] = 4;
        let (_state, bad) = record(Some(conflict));
        constants[16] = 5;
        std::hint::black_box(&constants);
        host.execute(&[&bad]);
        assert_eq!(get(10), 2, "incompatible command ran");
        assert_eq!(get(28), 3 + iteration as u32, "neighbor was skipped");
        eprintln!(
            "incompatible state {conflict:#x}: dispatches={}, neighbors={}",
            get(10),
            get(28)
        );
    }

    // One ExecuteCommandLists call, with the same list submitted twice
    // around a title-memory update. Both reads must survive completion.
    constants[16] = 4;
    let change = host.new_list().unwrap();
    change.defer(move || unsafe {
        ((cb + 64) as *mut u32).write(5);
    });
    change.close();
    host.execute(&[&list, &change, &list]);
    assert_eq!(
        [get(10), get(13), get(14), get(28)],
        [4, 0x100001c, 0x200001c, 6]
    );
    eprintln!(
        "same batch replayed: {:#x}, {:#x}; neighbors {}",
        get(13),
        get(14),
        get(28)
    );

    // Two guest-visible heaps can be placed on exactly the same memory.
    // Their dynamic slots and CBV upload records belong to separate hosts.
    let mut state = crate::draw::State::default();
    state.set_heaps(&[h]);
    state.set_table(true, 1, heap.start);
    let probe = host.new_list().unwrap();
    let first = state.prepared_cbv_table(&probe, &used, read);
    let mut other: This = std::ptr::null();
    let other_desc = Description(
        HeapDesc {
            kind: 0,
            count: 16,
            flags: 1,
            nodes: 0,
        },
        Placement {
            address: heap.start,
            size: 16 * 32,
        },
    );
    assert_eq!(
        crate::descriptor::create_heap(std::ptr::null(), &other_desc.0, &HEAP_IIDS[0], &mut other),
        0
    );
    let other_host = crate::descriptor::heap(other).host().unwrap();
    let reserved = other_host.dynamic(first + 32).unwrap();
    state.set_heaps(&[other]);
    let second = state.prepared_cbv_table(&probe, &used, read);
    assert!(second >= reserved + first + 32);
    eprintln!("heaps sharing guest VA: independent CBV slots {first}, {second}");
    other_host.release(reserved, first + 32);

    // Change one occupied guest record between distinct lists prepared in
    // the same batch. Keep the earlier slot; only the later command drops.
    constants[16] = 5;
    let (_first_state, first_list) = record(None);
    let (_next_state, next_list) = record(None);
    let replace_record = host.new_list().unwrap();
    replace_record.defer_copy(
        (heap.start + 5 * 32) as *mut u8,
        (heap.start + 4 * 32) as *const u8,
        12,
    );
    replace_record.close();
    host.execute(&[&first_list, &replace_record, &next_list]);
    assert_eq!([get(10), get(15), get(28)], [5, 0x200001c, 8]);
    eprintln!(
        "slot changed while in flight: first={:#x}, total={}, neighbors={}",
        get(15),
        get(10),
        get(28)
    );

    // A recorded selector retains the guest heap allocation past the
    // application's final reference, through another submission.
    crate::descriptor::release(h);
    host.execute(&[&list]);
    assert_eq!([get(10), get(16), get(28)], [6, 0x100001c, 9]);
    eprintln!("guest heap released before replay: {:#x}", get(16));
    assert_eq!(crate::host::indexed_suppressed(true), 3);
    assert_eq!(crate::host::indexed_suppressed(false), 0);
    eprintln!("suppressed at submit: compute=3, graphics=0");
}
