//! Xbox clear commands on CPU-visible title buffers and host textures.
use super::{Rect, This, ensure_state, host_list, with_state};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum UavClear {
    Uint([u32; 4]),
    Float([f32; 4]),
}

#[derive(Clone, Copy)]
struct BufferClear {
    pattern: [u8; 16],
    pattern_len: usize,
    stride: u64,
    first: u64,
    count: u64,
}

impl BufferClear {
    fn new(desc: Option<&[u32; 10]>, values: UavClear) -> Option<Self> {
        let (format, first, count, stride, raw) = match desc {
            Some(d) if d[1] == 1 => (
                d[0],
                u64::from(d[2]) | u64::from(d[3]) << 32,
                u64::from(d[4]),
                u64::from(d[5]),
                d[8] & 1 != 0,
            ),
            Some(_) => return None,
            None => (39, 0, u64::MAX, 0, true),
        };
        let mut pattern = [0; 16];
        if raw || stride != 0 {
            // The guest's raw flag takes precedence over typed encoding.
            if raw && stride != 0 {
                return None;
            }
            let bits = match values {
                UavClear::Uint(v) => v[0],
                UavClear::Float(v) if raw => v[0].to_bits(),
                UavClear::Float(_) => return None,
            };
            pattern[..4].copy_from_slice(&bits.to_le_bytes());
            if !raw && (format != 0 || stride % 4 != 0) {
                return None;
            }
            return Some(Self {
                pattern,
                pattern_len: 4,
                stride: if raw { 4 } else { stride },
                first,
                count,
            });
        }

        use Channel::{Float, Integer, Snorm, Unorm};
        // DXGI formats ordered by component width, count and encoding.
        let (width, channels, encoding) = match format {
            2 => (4, 4, Float),
            3 | 4 => (4, 4, Integer),
            6 => (4, 3, Float),
            7 | 8 => (4, 3, Integer),
            16 => (4, 2, Float),
            17 | 18 => (4, 2, Integer),
            41 => (4, 1, Float),
            42 | 43 => (4, 1, Integer),
            10 => (2, 4, Float),
            11 => (2, 4, Unorm),
            12 | 14 => (2, 4, Integer),
            13 => (2, 4, Snorm),
            34 => (2, 2, Float),
            35 => (2, 2, Unorm),
            36 | 38 => (2, 2, Integer),
            37 => (2, 2, Snorm),
            54 => (2, 1, Float),
            56 => (2, 1, Unorm),
            57 | 59 => (2, 1, Integer),
            58 => (2, 1, Snorm),
            28 | 87 => (1, 4, Unorm),
            30 | 32 => (1, 4, Integer),
            31 => (1, 4, Snorm),
            49 => (1, 2, Unorm),
            50 | 52 => (1, 2, Integer),
            51 => (1, 2, Snorm),
            61 => (1, 1, Unorm),
            62 | 64 => (1, 1, Integer),
            63 => (1, 1, Snorm),
            _ => return None,
        };
        let pattern_len = width * channels;
        for (i, component) in pattern[..pattern_len].chunks_exact_mut(width).enumerate() {
            let channel = if format == 87 && i < 3 { 2 - i } else { i };
            let bits = match values {
                UavClear::Uint(v) => v[channel],
                UavClear::Float(v) => encoding.encode(v[channel], width)?,
            };
            component.copy_from_slice(&bits.to_le_bytes()[..width]);
        }
        Some(Self {
            pattern,
            pattern_len,
            stride: pattern_len as u64,
            first,
            count,
        })
    }

    fn ranges(self, bytes: u64, rects: &[Rect]) -> Vec<std::ops::Range<usize>> {
        let elements = bytes.min(isize::MAX as u64) / self.stride;
        let count = self.count.min(elements.saturating_sub(self.first));
        if count == 0 {
            return Vec::new();
        }
        // Clamping in elements guarantees both endpoints fit before
        // converting to byte offsets.
        let range = |left, right| {
            ((self.first + left) * self.stride) as usize
                ..((self.first + right) * self.stride) as usize
        };
        if rects.is_empty() {
            return std::iter::once(range(0, count)).collect();
        }
        rects
            .iter()
            .filter_map(|rect| {
                let left = (rect.left.max(0) as u64).min(count);
                let right = (rect.right.max(0) as u64).min(count);
                (left < right).then(|| range(left, right))
            })
            .collect()
    }
}

enum Channel {
    Float,
    Integer,
    Snorm,
    Unorm,
}

impl Channel {
    fn encode(&self, value: f32, width: usize) -> Option<u32> {
        let value = if value.is_subnormal() {
            0.0_f32.copysign(value)
        } else {
            value
        };
        let bits = width * 8;
        Some(match self {
            Self::Float if width == 2 => u32::from(half_bits(value)),
            Self::Float => value.to_bits(),
            Self::Integer => return None,
            Self::Unorm => {
                (f64::from(value.clamp(0.0, 1.0)) * f64::from((1u32 << bits) - 1)).round() as u32
            }
            Self::Snorm => {
                (f64::from(value.clamp(-1.0, 1.0)) * f64::from((1u32 << (bits - 1)) - 1)).round()
                    as i32 as u32
            }
        })
    }
}

/// D3D's conversion to a narrower float rounds towards zero.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let fraction = bits & 0x7f_ffff;
    let magnitude = match exponent {
        128 if fraction == 0 => 0x7c00,
        128 => 0x7e00 | (fraction >> 13) as u16,
        16.. => 0x7bff,
        -14..=15 => (((exponent + 15) as u16) << 10) | (fraction >> 13) as u16,
        -24..=-15 => ((0x80_0000 | fraction) >> (-exponent - 1)) as u16,
        _ => 0,
    };
    sign | magnitude
}

pub(super) extern "win64" fn uint(
    this: This,
    gpu: u64,
    cpu: u64,
    resource: This,
    values: *const [u32; 4],
    count: u32,
    rects: *const Rect,
) {
    // SAFETY: the title passes four values or a null pointer.
    if let Some(&values) = unsafe { values.as_ref() } {
        clear_uav(
            this,
            gpu,
            cpu,
            resource,
            UavClear::Uint(values),
            count,
            rects,
        );
    }
}

pub(super) extern "win64" fn float(
    this: This,
    gpu: u64,
    cpu: u64,
    resource: This,
    values: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    // SAFETY: the title passes four values or a null pointer.
    if let Some(&values) = unsafe { values.as_ref() } {
        clear_uav(
            this,
            gpu,
            cpu,
            resource,
            UavClear::Float(values),
            count,
            rects,
        );
    }
}

fn clear_uav(
    this: This,
    gpu: u64,
    cpu: u64,
    resource: This,
    values: UavClear,
    count: u32,
    rects: *const Rect,
) {
    use crate::descriptor::{self, View};
    if (count != 0 && rects.is_null()) || count as usize > isize::MAX as usize / size_of::<Rect>() {
        return;
    }
    let is_uav = |view: &View| matches!(view, View::Unordered(..) | View::Placed(true, ..));
    let (Some(list), Some(view)) = (
        host_list(this),
        descriptor::view(cpu)
            .filter(is_uav)
            .or_else(|| descriptor::view(gpu).filter(is_uav)),
    ) else {
        return;
    };
    let rects = if count == 0 {
        &[]
    } else {
        // SAFETY: the title passes count rectangles at a non-null address.
        unsafe { std::slice::from_raw_parts(rects, count as usize) }
    };
    match &view {
        View::Unordered(res, _, desc) if !resource.is_null() && resource as usize == *res => {
            if let Some((data, len)) = crate::resource::buffer_bytes(resource) {
                clear_buffer_uav(
                    list,
                    (data as usize, len),
                    desc.as_ref(),
                    false,
                    values,
                    rects,
                );
            } else {
                clear_texture_uav(
                    this,
                    list,
                    gpu,
                    resource,
                    &view,
                    desc.as_ref(),
                    values,
                    rects,
                );
            }
        }
        View::Placed(true, res, desc) if res[0] == 1 => {
            let data = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let len = u64::from(res[4]) | u64::from(res[5]) << 32;
            let words = desc
                .first_chunk::<10>()
                .expect("placed UAV starts with the standard description");
            clear_buffer_uav(list, (data as usize, len), Some(words), true, values, rects);
        }
        View::Placed(true, ..) => {
            if let Some(View::Unordered(texture, _, words)) =
                descriptor::host::placed_texture(&view)
            {
                clear_texture_uav(
                    this,
                    list,
                    gpu,
                    texture as This,
                    &view,
                    words.as_ref(),
                    values,
                    rects,
                );
            }
        }
        _ => (),
    }
}

fn clear_buffer_uav(
    list: &crate::host::HostList,
    (data, len): (usize, u64),
    desc: Option<&[u32; 10]>,
    placed: bool,
    values: UavClear,
    rects: &[Rect],
) {
    let Some(clear) = BufferClear::new(desc, values) else {
        return;
    };
    let ranges = clear.ranges(len, rects);
    if data == 0
        || ranges.is_empty()
        || ranges
            .iter()
            .any(|range| data.checked_add(range.end).is_none())
    {
        return;
    }
    let mut shadows = Vec::new();
    let pattern = &clear.pattern[..clear.pattern_len];
    for range in &ranges {
        let at = (data + range.start) as u64;
        // Materialize this view even if the first shader that uses it is
        // recorded later or on another command list.
        let _ = crate::shadow::find(at, range.len() as u64);
        crate::shadow::overlapping(at, (data + range.end) as u64, |shadow, start, end| {
            shadows.push((
                shadow.clone(),
                start..end,
                (start - at) as usize % pattern.len(),
            ));
        });
    }
    let largest = ranges
        .iter()
        .map(std::ops::Range::len)
        .max()
        .unwrap_or_default();
    let block = pattern.repeat(largest.min(1 << 20).div_ceil(pattern.len()));
    let ordered = clear_shadow_uavs(list, pattern, &block, &shadows);
    #[cfg(not(feature = "diagnostics"))]
    let _ = placed;
    #[cfg(feature = "diagnostics")]
    if crate::producers::word().is_some() || crate::producers::mesh_selector().is_some() {
        for range in &ranges {
            let start = (data + range.start) as u64;
            list.producers.buffer_clear(crate::producers::BufferClear {
                range: (start, range.len() as u64),
                view: (data as u64, clear.first, clear.count, clear.stride),
                placed,
                float: matches!(values, UavClear::Float(_)),
                format: desc.map(|d| d[0]),
                pattern: clear.pattern,
                pattern_len: clear.pattern_len,
                gpu_copy: if !ordered {
                    Some(false)
                } else if shadows
                    .iter()
                    .any(|(_, r, _)| r.start <= start && r.end >= start + range.len() as u64)
                {
                    Some(true)
                } else if shadows
                    .iter()
                    .any(|(_, r, _)| r.start < start + range.len() as u64 && r.end > start)
                {
                    None // Partial aliases; the precise word might be selected later.
                } else {
                    Some(false)
                },
            });
        }
    }
    list.defer(move || {
        for range in &ranges {
            let at = data + range.start;
            if crate::texture::committed(at as *const u8, range.len()) {
                for pos in (0..range.len()).step_by(block.len()) {
                    let n = block.len().min(range.len() - pos);
                    crate::host::copy_mapped((at + pos) as *mut u8, block.as_ptr(), n, true);
                }
            }
        }
    });
    if !ordered {
        for (shadow, range, _) in shadows {
            shadow.refresh(list, range.start, range.end - range.start);
        }
    }
}

/// The title uses its memory for deferred CPU copies. Existing host
/// shadows also need a GPU-ordered copy so earlier shaders see the old
/// bytes and later ones see the clear.
fn clear_shadow_uavs(
    list: &crate::host::HostList,
    pattern: &[u8],
    block: &[u8],
    shadows: &[(
        std::sync::Arc<crate::shadow::Shadow>,
        std::ops::Range<u64>,
        usize,
    )],
) -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    if shadows.is_empty() {
        return true;
    }
    let Some(host) = crate::host::host() else {
        return false;
    };
    let Some((upload, data)) = list.upload(host, (block.len() + pattern.len()) as u64) else {
        return false;
    };
    // SAFETY: the list owns the upload mapping. The extra pattern lets
    // aliases that start partway through an element copy at that phase.
    unsafe {
        std::ptr::copy_nonoverlapping(block.as_ptr(), data, block.len());
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), data.add(block.len()), pattern.len());
    }
    for (shadow, range, phase) in shadows {
        let len = range.end - range.start;
        shadow.refresh(list, range.start, len);
        let mut expected = pattern.to_vec();
        expected.rotate_left(*phase);
        list.defer_gpu_clear(
            shadow,
            range.start as usize,
            len as usize,
            expected.into_boxed_slice(),
        );

        let previous = list.bound_buffer_state(&shadow.buffer);
        list.buffer_state(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
        // SAFETY: the ranges fit their live buffers, tracked in copy states.
        unsafe {
            for pos in (0..len).step_by(block.len()) {
                list.list.CopyBufferRegion(
                    &shadow.buffer,
                    shadow.offset(range.start) + pos,
                    &upload,
                    *phase as u64,
                    (len - pos).min(block.len() as u64),
                );
            }
        }
        if let Some(state) = previous {
            list.buffer_state(&shadow.buffer, state);
        }
    }
    true
}

fn clear_texture_uav(
    this: This,
    list: &crate::host::HostList,
    gpu: u64,
    resource: This,
    view: &crate::descriptor::View,
    words: Option<&[u32; 10]>,
    values: UavClear,
    rects: &[Rect],
) {
    let (Some((texture, _)), Some(host), Some((heap, index))) = (
        crate::resource::host_unordered(resource, words),
        crate::host::host(),
        crate::descriptor::clear_uav_slot(gpu, view),
    ) else {
        return;
    };
    let Some(desc) = crate::descriptor::host::texture_uav_desc(resource, words) else {
        return;
    };
    const UNORDERED_ACCESS: u32 = 0x8;
    ensure_state(list, resource, UNORDERED_ACCESS);
    // SAFETY: Rect and the host's RECT are four i32 fields in C order.
    let rects = unsafe { std::slice::from_raw_parts(rects.as_ptr().cast(), rects.len()) };
    host.with_clear_uav(list, &heap.heap, texture, desc.as_ref(), |cpu| {
        let gpu = heap.gpu(index);
        // SAFETY: a live texture in UAV state and matching UAV descriptors,
        // with the shader-visible heap bound for this call.
        unsafe {
            match values {
                UavClear::Uint(v) => list
                    .list
                    .ClearUnorderedAccessViewUint(gpu, cpu, texture, &v, rects),
                UavClear::Float(v) => list
                    .list
                    .ClearUnorderedAccessViewFloat(gpu, cpu, texture, &v, rects),
            }
        }
    });
    with_state(this, |_, state| state.invalidate_host_heaps());
    crate::texture::host_written(resource);
}
