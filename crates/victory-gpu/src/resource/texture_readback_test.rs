use super::*;
use std::ptr::null;

#[repr(C)]
struct Location(This, u32, PlacedFootprint);

type CopyTexture = extern "win64" fn(This, &Location, u32, u32, u32, &Location, *const [u32; 6]);
type CopyBuffer = extern "win64" fn(This, This, u64, This, u64, u64);
type Execute = extern "win64" fn(This, u32, *const This);

#[test]
#[ignore = "requires a D3D12 host"]
fn title_texture_readbacks_preserve_neighbor_bytes_and_replay_order() {
    crate::host::host().expect("requires D3D12");
    let make = |desc, state| {
        let mut r = null();
        assert_eq!(
            create_committed(
                null(),
                &PROPERTIES,
                0,
                &desc,
                state,
                null(),
                &RESOURCE_IIDS[0],
                &mut r
            ),
            S_OK
        );
        r
    };
    let image = make(texture(5, 3, 2, 1), 0x400); // RGBA32F, arbitrary bit patterns
    let meter = make(texture(5, 1, 2, 1), 0x400);
    let single = make(texture(1, 1, 2, 1), 0x400);
    let (source, dst, observer) = (
        make(buffer(1024), 0),
        make(buffer(1536), 0),
        make(buffer(1536), 0),
    );
    let iid = &PAGEABLE_IIDS[1];
    let mut queue = null();
    assert_eq!(
        crate::queue::create(null(), [0u32; 4].as_ptr().cast(), iid, &mut queue),
        S_OK
    );
    let make_list = || {
        let (mut a, mut l) = (null(), null());
        assert_eq!(crate::list::create_allocator(null(), 0, iid, &mut a), S_OK);
        assert_eq!(
            crate::list::create_list(null(), 0, 0, a, null(), iid, &mut l),
            S_OK
        );
        (a, l)
    };
    let (a, write) = make_list();
    let (b, read) = make_list();
    let (c, observe) = make_list();
    let close = |l| assert_eq!(slot::<extern "win64" fn(This) -> HRESULT>(l, 0x48)(l), S_OK);
    let f = PlacedFootprint {
        offset: 0,
        format: 2,
        width: 5,
        height: 3,
        depth: 1,
        row_pitch: 256,
    };
    let src = Location(image, 0, f);
    slot::<CopyTexture>(write, 0x80)(write, &src, 0, 0, 0, &Location(source, 1, f), null());
    for (target, width, offset) in [(meter, 5, 0), (single, 1, 576)] {
        let footprint = PlacedFootprint {
            width,
            height: 1,
            ..f
        };
        slot::<CopyTexture>(write, 0x80)(
            write,
            &Location(target, 0, f),
            0,
            0,
            0,
            &Location(
                source,
                1,
                PlacedFootprint {
                    offset,
                    ..footprint
                },
            ),
            null(),
        );
    }
    close(write);
    let copy = slot::<CopyTexture>(read, 0x80);
    // Two exposure-sized reads, and two partial rows with both origins nonzero.
    copy(
        read,
        &Location(dst, 1, PlacedFootprint { height: 1, ..f }),
        0,
        0,
        0,
        &Location(meter, 0, f),
        null(),
    );
    copy(
        read,
        &Location(
            dst,
            1,
            PlacedFootprint {
                offset: 512,
                width: 1,
                height: 1,
                ..f
            },
        ),
        0,
        0,
        0,
        &Location(single, 0, f),
        null(),
    );
    copy(
        read,
        &Location(
            dst,
            1,
            PlacedFootprint {
                offset: 801,
                row_pitch: 111,
                ..f
            },
        ),
        1,
        1,
        0,
        &src,
        &[2, 1, 0, 4, 3, 1],
    );
    // The requested range extends past the end of dst. No partial write.
    copy(
        read,
        &Location(dst, 1, PlacedFootprint { offset: 1530, ..f }),
        0,
        0,
        0,
        &src,
        &[0, 0, 0, 1, 1, 1],
    );
    close(read);
    slot::<CopyBuffer>(observe, 0x78)(observe, observer, 0, dst, 0, 1536);
    close(observe);
    let host_bytes = |r| {
        let (p, n) = buffer_bytes(r).unwrap();
        unsafe { std::slice::from_raw_parts_mut(p, n as usize) }
    };
    let input = host_bytes(source);
    let output = host_bytes(dst);
    let seen = host_bytes(observer);
    let run = slot::<Execute>(queue, 0x50);
    for seed in [0x11u8, 0x79, 0x11] {
        for (i, v) in input.iter_mut().enumerate() {
            *v = seed.wrapping_add((i % 251) as u8);
        }
        output.fill(0xef);
        seen.fill(0xa4);
        let mut expected = vec![0xef; 1536];
        expected[..80].copy_from_slice(&input[..80]);
        expected[512..528].copy_from_slice(&input[576..592]);
        for (y, at) in [928, 1039].into_iter().enumerate() {
            let start = (y + 1) * 256 + 32;
            expected[at..at + 32].copy_from_slice(&input[start..start + 32]);
        }
        let runs = [write, read, observe, read, observe];
        run(queue, runs.len() as u32, runs.as_ptr());
        assert_eq!(output, &expected);
        assert_eq!(seen, &expected, "the following list sees the fenced result");
    }
    eprintln!(
        "GPU published 5x1 and 1x1 RGBA32F plus a boxed copy with 111-byte guest pitch; 1536 destination and 1536 observer bytes matched across 3 submissions, each replaying the readback list twice"
    );
    for obj in [
        write, read, observe, a, b, c, queue, image, meter, single, source, dst, observer,
    ] {
        slot::<Release>(obj, 0x10)(obj);
    }
}
