use super::*;
use crate::list::{
    ALLOCATOR_IIDS, COPY_DEST, COPY_SOURCE, LIST_FUNCTIONS, LIST_IIDS, NEWER_LIST_FUNCTIONS,
    UAV_BARRIER, barrier, close, copy_buffer_region, create_allocator, create_list,
};
use crate::object::{S_OK, slot};
use crate::resource::PAGEABLE_IIDS;

fn buffer_uav_desc(format: u32, first: u64, count: u32, stride: u32, flags: u32) -> [u32; 10] {
    [
        format,
        1,
        first as u32,
        (first >> 32) as u32,
        count,
        stride,
        0,
        0,
        flags,
        0,
    ]
}

fn rect(left: i32, top: i32, right: i32, bottom: i32) -> Rect {
    Rect {
        left,
        top,
        right,
        bottom,
    }
}

#[test]
fn uav_slots_are_stable() {
    for table in [&LIST_FUNCTIONS, &NEWER_LIST_FUNCTIONS] {
        assert_eq!(table.0[53], uint as *const ());
        assert_eq!(table.0[54], float as *const ());
    }
}

#[test]
fn uav_buffer_ranges_clamp_to_the_resource_and_view() {
    let raw = buffer_uav_desc(39, 2, 5, 0, 1);
    let clear = BufferClear::new(Some(&raw), UavClear::Uint([0x12345678, 1, 2, 3])).unwrap();
    assert_eq!(
        &clear.pattern[..clear.pattern_len],
        &[0x78, 0x56, 0x34, 0x12]
    );
    assert_eq!(clear.ranges(29, &[]), vec![8..28]);
    assert_eq!(
        clear.ranges(
            24,
            &[
                rect(-5, 50, 1, 50),
                rect(2, 0, i32::MAX, 1),
                rect(4, 0, 3, 1)
            ]
        ),
        vec![8..12, 16..24],
    );
    let beyond = buffer_uav_desc(3, u64::MAX, u32::MAX, 0, 0);
    assert!(
        BufferClear::new(Some(&beyond), UavClear::Uint([0; 4]))
            .unwrap()
            .ranges(u64::MAX, &[])
            .is_empty()
    );
    let default = BufferClear::new(None, UavClear::Uint([4, 3, 2, 1])).unwrap();
    assert_eq!(default.ranges(18, &[]), vec![0..16]);
}

#[test]
fn uav_buffer_component_encodings() {
    let check = |format, values, expected: &[u8]| {
        let desc = buffer_uav_desc(format, 1, 4, 0, 0);
        let clear = BufferClear::new(Some(&desc), values).unwrap();
        assert_eq!(&clear.pattern[..clear.pattern_len], expected);
        assert_eq!(
            clear.ranges(128, &[rect(1, 0, 3, 1)]),
            vec![2 * expected.len()..4 * expected.len()],
        );
    };
    check(42, UavClear::Uint([0x3412, 7, 8, 9]), &[0x12, 0x34, 0, 0]);
    check(
        3,
        UavClear::Uint([1, 2, 3, 4]),
        &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
    );
    check(
        41,
        UavClear::Float([-2.0, 0.0, 0.0, 0.0]),
        &(-2.0_f32).to_le_bytes(),
    );
    check(
        41,
        UavClear::Uint([0x7fc00042, 0, 0, 0]),
        &[0x42, 0, 0xc0, 0x7f],
    );
    check(
        28,
        UavClear::Float([-1.0, 0.5, 2.0, f32::NAN]),
        &[0, 128, 255, 0],
    );
    check(
        28,
        UavClear::Uint([0x100, 0x180, 0x1ff, 0x142]),
        &[0, 128, 255, 66],
    );
    check(87, UavClear::Float([0.0, 0.5, 1.0, 0.0]), &[255, 128, 0, 0]);
    check(58, UavClear::Float([-1.0, 0.0, 0.0, 0.0]), &[1, 128]);
    check(54, UavClear::Float([1.5, 0.0, 0.0, 0.0]), &[0, 62]);
    let structured = buffer_uav_desc(0, 1, 2, 20, 0);
    let clear = BufferClear::new(Some(&structured), UavClear::Uint([3, 7, 8, 9])).unwrap();
    assert_eq!(&clear.pattern[..clear.pattern_len], &[3, 0, 0, 0]);
    assert_eq!(clear.ranges(80, &[]), vec![20..60]);
    assert!(BufferClear::new(Some(&structured), UavClear::Float([0.0; 4])).is_none());
    assert!(
        BufferClear::new(
            Some(&buffer_uav_desc(42, 0, 1, 0, 0)),
            UavClear::Float([0.0; 4])
        )
        .is_none()
    );
}

mod uav_live {
    use super::*;
    use crate::descriptor;
    use crate::resource;
    use std::ptr::null;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
    use windows::core::Interface;

    struct Guest(This);

    impl Drop for Guest {
        fn drop(&mut self) {
            slot::<extern "win64" fn(This) -> u32>(self.0, 0x10)(self.0);
        }
    }

    fn guest_list() -> (Guest, Guest, Guest) {
        let (mut list, mut queue, mut allocator) = (null(), null(), null());
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(
                null(),
                queue_desc.as_ptr().cast(),
                &PAGEABLE_IIDS[0],
                &mut queue
            ),
            S_OK
        );
        assert_eq!(
            create_allocator(null(), 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        assert_eq!(
            create_list(null(), 0, 0, allocator, null(), &LIST_IIDS[1], &mut list),
            S_OK
        );
        (Guest(list), Guest(queue), Guest(allocator))
    }

    fn submit(list: This, queue: This) {
        assert_eq!(close(list), S_OK);
        slot::<extern "win64" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
    }

    fn resource(dimension: u32, width: u64, height: u32, format: u32) -> Guest {
        let properties = [1u32, 0, 0, 1, 1];
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION(dimension as i32),
            Width: width,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT(format as i32),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: if dimension == 1 {
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR
            } else {
                D3D12_TEXTURE_LAYOUT_UNKNOWN
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut out = null();
        assert_eq!(
            resource::create_committed(
                null(),
                properties.as_ptr().cast(),
                0,
                (&raw const desc).cast(),
                0,
                null(),
                &PAGEABLE_IIDS[0],
                &mut out
            ),
            S_OK
        );
        Guest(out)
    }

    type Clear<T> = extern "win64" fn(This, u64, u64, This, *const [T; 4], u32, *const Rect);

    fn readback(host: &crate::host::Host, width: u64) -> ID3D12Resource {
        let mut buffer = None;
        // SAFETY: a readback resource description and a live device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: width,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .unwrap();
        }
        buffer.unwrap()
    }

    #[test]
    fn uav_raw_and_placed_buffers_clear_at_submission() {
        if crate::host::host().is_none() {
            eprintln!("skipping buffer UAV submission test without a host GPU");
            return;
        }
        let buffer = resource(1, 32, 1, 0);
        let copied = resource(1, 32, 1, 0);
        let (data, _) = resource::buffer_bytes(buffer.0).unwrap();
        let shadow = crate::shadow::find(data as u64, 32).unwrap();
        let (list, queue, _allocator) = guest_list();
        let raw = buffer_uav_desc(39, 1, 6, 0, 1);
        let mut cpu = [0u32; 8];
        let cpu = cpu.as_mut_ptr() as u64;
        descriptor::create_uav(null(), buffer.0, null(), &raw, cpu);
        let clear = slot::<Clear<u32>>(list.0, 0x1a8);
        clear(list.0, 0, cpu, buffer.0, &[10, 11, 12, 13], 0, null());
        copy_buffer_region(list.0, copied.0, 0, buffer.0, 0, 32);
        clear(list.0, 0, cpu, buffer.0, &[20; 4], 1, &rect(1, 0, 3, 1));

        let mut placed = [99u32; 6];
        let at = placed.as_mut_ptr() as u64;
        let mut res = [0u32; 14];
        res[0] = 1;
        res[4] = 24;
        let mut desc = [0; 16];
        desc[..10].copy_from_slice(&raw);
        desc[10] = at as u32;
        desc[11] = (at >> 32) as u32;
        let mut gpu = [0u32; 8];
        let gpu = gpu.as_mut_ptr() as u64;
        descriptor::create_placed_uav(null(), &res, &desc, gpu);
        clear(list.0, gpu, 0, null(), &[7; 4], 1, &rect(2, 0, 100, 1));
        let read = |resource| {
            let (data, _) = resource::buffer_bytes(resource).unwrap();
            // SAFETY: the fixture has eight initialized dwords.
            unsafe { data.cast::<[u32; 8]>().read_unaligned() }
        };
        assert_eq!(read(buffer.0), [0; 8]);
        assert_eq!(placed, [99; 6]);
        submit(list.0, queue.0);
        assert_eq!(read(copied.0), [0, 10, 10, 10, 10, 10, 10, 0]);
        let expected = [0, 10, 20, 20, 10, 10, 10, 0];
        assert_eq!(read(buffer.0), expected);
        assert_eq!(placed, [99, 99, 99, 7, 7, 7]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the list has finished and the shadow holds this buffer.
        unsafe {
            shadow.buffer.Map(0, None, Some(&mut mapped)).unwrap();
            assert_eq!(mapped.cast::<[u32; 8]>().read_unaligned(), expected);
            shadow.buffer.Unmap(0, Some(&D3D12_RANGE::default()));
        }
    }

    #[test]
    fn uav_buffer_shadow_clears_in_gpu_order() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (list, queue, _allocator) = guest_list();
        let native = host_list(list.0).unwrap();
        let buffer = resource(1, 16, 1, 0);
        let (data, _) = resource::buffer_bytes(buffer.0).unwrap();
        let shadow = crate::shadow::find(data as u64, 16).unwrap();
        let back = readback(host, 32);
        let raw = buffer_uav_desc(39, 0, 4, 0, 1);
        let mut descriptor = [0u32; 8];
        let cpu = descriptor.as_mut_ptr() as u64;
        descriptor::create_uav(null(), buffer.0, null(), &raw, cpu);
        let clear = slot::<Clear<u32>>(list.0, 0x1a8);
        clear(list.0, 0, cpu, buffer.0, &[12; 4], 0, null());
        native.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
        // SAFETY: the live, 16-byte shadow has been cleared by this list.
        unsafe {
            native
                .list
                .CopyBufferRegion(&back, 0, &shadow.buffer, 0, 16)
        };
        clear(list.0, 0, cpu, buffer.0, &[23; 4], 0, null());
        // A later GPU write restores the original zero only in word 2.
        // Comparing against the pre-clear guest snapshot would lose it.
        let (zero, mapped) = native.upload(host, 4).unwrap();
        shadow.gpu_writes(native, data as u64 + 8, 4);
        native.buffer_state(&shadow.buffer, COPY_DEST);
        unsafe {
            mapped.cast::<u32>().write_unaligned(0);
            native.list.CopyBufferRegion(&shadow.buffer, 8, &zero, 0, 4);
        }
        native.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
        // SAFETY: the other half of the same 32-byte readback buffer.
        unsafe {
            native
                .list
                .CopyBufferRegion(&back, 16, &shadow.buffer, 0, 16)
        };
        submit(list.0, queue.0);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: both copies finished and the buffer contains eight dwords.
        unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            assert_eq!(
                mapped.cast::<[u32; 8]>().read_unaligned(),
                [12, 12, 12, 12, 23, 23, 0, 23]
            );
            back.Unmap(0, Some(&D3D12_RANGE::default()));
            assert_eq!(data.cast::<[u32; 4]>().read_unaligned(), [23, 23, 0, 23]);
        }
    }

    #[test]
    fn uav_typed_buffers_keep_bytes_outside_the_rect() {
        if crate::host::host().is_none() {
            eprintln!("skipping typed UAV submission test without a host GPU");
            return;
        }
        let (list, queue, _allocator) = guest_list();
        let cases = [
            (
                42,
                UavClear::Uint([0x2211, 0, 0, 0]),
                &[0x11, 0x22, 0, 0][..],
            ),
            (
                3,
                UavClear::Uint([1, 2, 3, 4]),
                &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
            ),
            (41, UavClear::Float([1.0; 4]), &[0, 0, 128, 63]),
            (
                28,
                UavClear::Float([0.0, 0.5, 1.0, 0.25]),
                &[0, 128, 255, 64],
            ),
        ];
        let mut buffers = Vec::new();
        for (format, values, expected) in cases {
            let size = 4 * expected.len();
            let buffer = resource(1, size as u64, 1, 0);
            let (data, _) = resource::buffer_bytes(buffer.0).unwrap();
            // SAFETY: the committed fixture owns size bytes.
            unsafe { data.write_bytes(0xcd, size) };
            let desc = buffer_uav_desc(format, 1, 2, 0, 0);
            let mut cpu = [0u32; 8];
            let cpu = cpu.as_mut_ptr() as u64;
            descriptor::create_uav(null(), buffer.0, null(), &desc, cpu);
            let region = rect(1, 0, 2, 1);
            match values {
                UavClear::Uint(v) => {
                    slot::<Clear<u32>>(list.0, 0x1a8)(list.0, 0, cpu, buffer.0, &v, 1, &region)
                }
                UavClear::Float(v) => {
                    slot::<Clear<f32>>(list.0, 0x1b0)(list.0, 0, cpu, buffer.0, &v, 1, &region)
                }
            }
            buffers.push((buffer, data, expected));
        }
        submit(list.0, queue.0);
        for (_buffer, data, expected) in buffers {
            let len = expected.len();
            // SAFETY: the fixture still owns the four elements.
            let bytes = unsafe { std::slice::from_raw_parts(data, 4 * len) };
            assert_eq!(&bytes[2 * len..3 * len], expected);
            assert!(
                bytes[..2 * len]
                    .iter()
                    .chain(&bytes[3 * len..])
                    .all(|&b| b == 0xcd)
            );
        }
    }

    #[test]
    fn uav_texture_uint_and_float_readback() {
        if !descriptor::host::placed_fixture_process() {
            return;
        }
        let Some(host) = crate::host::host() else {
            eprintln!("skipping texture UAV readback without a host GPU");
            return;
        };
        host.log_messages();
        let before = host.debug_issues.load(std::sync::atomic::Ordering::Relaxed);
        let (list, queue, _allocator) = guest_list();
        let owned = [resource(3, 4, 4, 42), resource(3, 4, 4, 41)];
        let placed_memory = Box::new([0u32; 64]);
        let address = placed_memory.as_ptr() as u64;
        let placed_desc = [3, 0, 0, 0, 4, 0, 4, 1 | 1 << 16, 42, 1, 0, 0, 4, 0];
        let placed = resource::texture_at(address, &placed_desc).unwrap();
        let images = [owned[0].0, owned[1].0, placed];
        let heap_desc = [0u32, 3, 1, 0, 0, 0, 0, 0];
        let mut heap = null();
        assert_eq!(
            descriptor::create_heap(
                null(),
                heap_desc.as_ptr().cast(),
                &PAGEABLE_IIDS[0],
                &mut heap
            ),
            S_OK
        );
        let heap = Guest(heap);
        let mut cpu = [[0u32; 8]; 3];
        let center = rect(1, 1, 3, 3);
        for (i, &image) in images.iter().enumerate() {
            let format = [42, 41, 42][i];
            let words = [format, 4, 0, 0, 0, 0, 0, 0, 0, 0];
            let cpu = cpu[i].as_mut_ptr() as u64;
            let gpu = descriptor::heap(heap.0).start + i as u64 * 32;
            if i == 2 {
                let mut view = [0; 16];
                view[1] = 4;
                view[10] = address as u32;
                view[11] = (address >> 32) as u32;
                descriptor::create_placed_uav(null(), &placed_desc, &view, cpu);
            } else {
                descriptor::create_uav(null(), image, null(), &words, cpu);
            }
            descriptor::copy_simple(null(), 1, gpu, cpu, 0);
            if i != 1 {
                let argument = if i == 2 { null() } else { image };
                let clear = slot::<Clear<u32>>(list.0, 0x1a8);
                clear(list.0, gpu, cpu, argument, &[3 + i as u32; 4], 0, null());
                barrier(list.0, 1, &UAV_BARRIER);
                clear(list.0, gpu, cpu, argument, &[9 + i as u32; 4], 1, &center);
            } else {
                let clear = slot::<Clear<f32>>(list.0, 0x1b0);
                clear(list.0, gpu, 0, image, &[-1.25; 4], 0, null());
                barrier(list.0, 1, &UAV_BARRIER);
                clear(list.0, gpu, cpu, image, &[6.5; 4], 1, &center);
            }
        }

        let back = readback(host, 3072);
        let host_list = host_list(list.0).unwrap();
        // D3D12_TEXTURE_COPY_LOCATION's two union variants share this layout.
        #[repr(C)]
        struct Location {
            resource: *mut std::ffi::c_void,
            kind: D3D12_TEXTURE_COPY_TYPE,
            footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        }
        for (i, &image) in images.iter().enumerate() {
            let texture = resource::host_resource(image).unwrap();
            ensure_state(host_list, image, COPY_SOURCE);
            let src = Location {
                resource: texture.as_raw(),
                kind: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                footprint: Default::default(),
            };
            let dst = Location {
                resource: back.as_raw(),
                kind: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 1024 * i as u64,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT([42, 41, 42][i]),
                        Width: 4,
                        Height: 4,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            };
            // SAFETY: valid copy locations and an open list.
            unsafe {
                host_list.list.CopyTextureRegion(
                    (&raw const dst).cast(),
                    0,
                    0,
                    0,
                    (&raw const src).cast(),
                    None,
                );
            }
        }
        submit(list.0, queue.0);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the guest submission waited for both copies. Each row is 256 bytes.
        unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            let words = std::slice::from_raw_parts(mapped.cast::<u32>(), 768);
            for (i, (outer, inner)) in [(3, 9), ((-1.25_f32).to_bits(), 6.5_f32.to_bits()), (5, 11)]
                .into_iter()
                .enumerate()
            {
                for y in 0..4 {
                    for x in 0..4 {
                        let expected = if (1..3).contains(&x) && (1..3).contains(&y) {
                            inner
                        } else {
                            outer
                        };
                        assert_eq!(
                            words[i * 256 + y * 64 + x],
                            expected,
                            "texture {i}, pixel ({x}, {y})"
                        );
                    }
                }
            }
            back.Unmap(0, Some(&D3D12_RANGE::default()));
        }
        assert_eq!(
            host.debug_issues.load(std::sync::atomic::Ordering::Relaxed),
            before
        );
    }
}
