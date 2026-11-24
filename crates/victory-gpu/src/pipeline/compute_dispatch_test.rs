//! Verify the contents a dependent GPU command reads on first use of a compute PSO.
use super::*;
use crate::host::{BUFFER_READ, BUFFER_WRITE, host};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT_R32_TYPELESS};

#[test]
fn first_compute_dispatch_feeds_the_next_dispatch() {
    // The wait budget and device are process-wide. Start with a cold PSO
    // and a zero budget even when other GPU tests have already run.
    const CHILD: &str = "VICTORY_TEST_FIRST_DISPATCH_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let name =
            "pipeline::compute_dispatch_test::first_compute_dispatch_feeds_the_next_dispatch";
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([name, "--exact", "--nocapture"])
            .env(CHILD, "1")
            .env("VICTORY_PSO_WAIT_MS", "0")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "{stdout}{stderr}"
        );
        eprint!("{stderr}");
        return;
    }
    let Some(host) = host() else {
        eprintln!("skipped compute dependency test: no host device");
        return;
    };
    let code = victory_shader::hostlib::compile_hlsl(
        "[numthreads(1,1,1)] void main() { RWByteAddressBuffer b = ResourceDescriptorHeap[0]; b.Store(0, 2u * b.Load(0)); }",
        "main", "cs_6_6",
    ).expect("compile the producer and consumer");
    let cs = shader("compute", &code);
    cs.translated.get_or_init(|| {
        Some(Translated {
            host: victory_shader::emit::HostShader {
                stage: victory_shader::HwStage::Cs,
                dxil: code,
                user_data: vec![],
                samplers: vec![],
                static_samplers: vec![],
                root_cbvs: vec![],
                root_srvs: vec![],
                root_uavs: vec![],
                indexed_reads: vec![],
                stride_table: false,
            },
            used: vec![],
        })
    });
    let list = host.new_list().unwrap();
    let (buffer, data) = host.shared_buffer(256).expect("mapped UAV");
    let heap: ID3D12DescriptorHeap = unsafe {
        host.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap()
    };
    // SAFETY: the mapping is aligned and owned by the CPU until submission.
    // The buffer and heap remain alive until the submission fence completes.
    unsafe {
        data.cast::<u32>().write(7);
        host.device.CreateUnorderedAccessView(
            &buffer,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: 64,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            }),
            heap.GetCPUDescriptorHandleForHeapStart(),
        );
        list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
    }
    let mut pso = std::ptr::null();
    assert_eq!(
        hand_out(vec![cs], None, &PIPELINE_IIDS[0], &mut pso),
        crate::object::S_OK
    );
    let mut state = crate::draw::State::default();
    state.set_pipeline(pso);
    list.buffer_state(&buffer, BUFFER_WRITE);
    assert!(
        state.dispatch(&list, 1, 1, 1),
        "first dispatch was discarded"
    );
    // Complete the first UAV write before the second dispatch reads its value.
    list.buffer_state(&buffer, BUFFER_READ);
    list.buffer_state(&buffer, BUFFER_WRITE);
    assert!(
        state.dispatch(&list, 1, 1, 1),
        "dependent dispatch was discarded"
    );
    list.close();
    host.execute(&[&list]);
    // SAFETY: host.execute waited for the GPU and the mapping remains live.
    let result = unsafe { data.cast::<u32>().read() };
    assert_eq!(result, 28, "both consecutive GPU writes must be visible");
    assert_eq!(release(pso), 0);
}

// A stripped-down four-user-SGPR program. It reads a byte offset, a value,
// and an entire root UAV address from the packed arguments and stores one word.
fn spilled_root_store() -> Translated {
    use victory_shader::native::{ComputeIo, StageIo};
    use victory_shader::{Container, FourCC, HwStage, NativeProgram, Part, RootParam, RootType};
    let words: [u32; 13] = [
        0x8703ff01, 0xff, 0xbe820300, // masked s[0:1] copied to s[2:3]
        0xf4080201, 0xfa000000, // s[8:11] = offset, value, low VA, high VA
        0xbe8c030a, 0xbe8d030b, 0xbe8e03c1, 0xbe8f0380, // raw V# in s[12:15]
        0x7e000209, // v0 = s9 (the value)
        0xe0700000, 0x08030000, // buffer_store_dword v0, off, s[12:15], s8
        0xbf810000,
    ];
    let param = |root, slot, user_data| RootParam {
        root,
        slot,
        user_data,
        sgprs: 2,
        base_register: 0,
        table_offset_dwords: 0,
    };
    let used = victory_shader::UsedResource {
        root: RootType::Descriptor,
        slot: 1,
        user_data: 18,
        kind: victory_shader::ResourceType::Uav,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        table_offset_dwords: 0,
    };
    let n = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 1,
        layout: vec![
            param(RootType::Constants, 0, 16),
            param(RootType::Descriptor, 1, 18),
        ],
        used: vec![used],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let sig: [u32; 18] = [2, 2, 24, 0, 72, 0, 1, 0, 48, 4, 0, 60, 0, 0, 2, 0, 0, 0];
    let rts: Vec<_> = sig.iter().flat_map(|w| w.to_le_bytes()).collect();
    let c = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![Part {
            fourcc: FourCC::RTS0,
            offset: 0,
            data: &rts,
        }],
    };
    let p = victory_shader::ir::build(&n.code, false, 1).unwrap();
    let st = victory_shader::structure::structure(&p).unwrap();
    let io = victory_shader::io::raise(&p, &n).unwrap();
    let r = victory_shader::resource::track_shader(&p, &n, &c, Some(&st)).unwrap();
    let host =
        victory_shader::emit::emit(&c, &n, &p, &st, &io, &r, victory_shader::emit::Target::ANY)
            .unwrap();
    assert_eq!(host.root_uavs, [1]);
    Translated { host, used: n.used }
}

#[test]
fn spilled_root_uav_writes_overlapping_views_with_guest_offsets_and_bounds() {
    let Some(host) = host() else {
        eprintln!("skipped: no host device");
        return;
    };
    let trans = spilled_root_store();
    let cs = shader("compute", &trans.host.dxil);
    cs.translated.get_or_init(|| Some(trans));
    let mut pso = std::ptr::null();
    assert_eq!(
        hand_out(vec![cs], None, &PIPELINE_IIDS[0], &mut pso),
        crate::object::S_OK
    );

    // Console heap description followed by an empty placement, as in the title.
    let desc = [0u32, 8, 1, 0, 0, 0, 0, 0];
    let mut heap = std::ptr::null();
    assert_eq!(
        crate::descriptor::create_heap(
            std::ptr::null(),
            desc.as_ptr().cast(),
            &crate::resource::PAGEABLE_IIDS[0],
            &mut heap
        ),
        crate::object::S_OK
    );
    let mem = crate::resource::Memory::zeroed(128, 64).unwrap();
    let address = mem.address();
    let owner = address as usize;
    crate::shadow::register(owner, address, 64);
    let shadow = crate::shadow::find(address, 64).unwrap();
    let mut mapped = std::ptr::null_mut();
    // SAFETY: a mapped 64-KiB host allocation backs this 64-byte guest resource.
    // The next dword is a guard outside both requested views.
    unsafe {
        shadow.buffer.Map(0, None, Some(&mut mapped)).unwrap();
        mapped.cast::<u32>().add(16).write(0xcafef00d);
    }
    let list = host.new_list().unwrap();
    let mut state = crate::draw::State::default();
    state.set_pipeline(pso);
    state.set_heaps(&[heap]);
    for (base, offset, value) in [(16, 4, 7), (32, 4, 11), (16, 20, 29), (32, 32, 55)] {
        state.set_view(true, 1, address + base);
        state.set_constants(true, 0, &[offset, value], 0);
        assert!(state.dispatch(&list, 1, 1, 1));
        // Order successive writes to overlapping views through the same real buffer.
        list.buffer_state(&shadow.buffer, BUFFER_READ);
    }
    list.close();
    host.execute(&[&list]);
    let mut expected = [0u32; 16];
    expected[5] = 7;
    expected[9] = 29; // the +32 and +16 views addressed the same word
    // SAFETY: execute waited for the fence and copied GPU writes back to title memory.
    unsafe {
        assert_eq!(
            std::slice::from_raw_parts(address as *const u32, 16),
            expected
        );
        assert_eq!(mapped.cast::<u32>().add(16).read(), 0xcafef00d);
        shadow.buffer.Unmap(0, None);
    }
    crate::shadow::unregister(owner, address);
    drop(state);
    drop(list);
    let release_heap: extern "win64" fn(crate::object::This) -> u32 =
        crate::object::slot(heap, 2 * std::mem::size_of::<usize>());
    assert_eq!(release_heap(heap), 0);
    assert_eq!(release(pso), 0);
}
