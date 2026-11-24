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
