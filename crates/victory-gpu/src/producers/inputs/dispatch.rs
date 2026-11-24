//! Decode the same one-argument command records the host submits.
use crate::host::{BUFFER_READ, HostList};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

type BufferAt<'a> = (&'a ID3D12Resource, u64);

pub(crate) enum Dispatch<'a> {
    Direct([u32; 3]),
    Indirect {
        guest_kind: u32,
        stride: u32,
        max: u32,
        args: BufferAt<'a>,
        count: Option<BufferAt<'a>>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Plan {
    Direct([u32; 3]),
    Indirect {
        guest_kind: u32,
        stride: u32,
        max: u32,
        counted: bool,
    },
}

impl Dispatch<'_> {
    /// Capture signature use even when the chosen output record belongs to
    /// another dispatch. The guest encodings 2 and 105 share one host form.
    pub(crate) fn note_signature(&self) {
        if let Self::Indirect {
            guest_kind,
            stride,
            max,
            count,
            ..
        } = self
        {
            use std::sync::{Mutex, OnceLock};
            static SEEN: OnceLock<Mutex<std::collections::HashSet<(u32, u32, bool)>>> =
                OnceLock::new();
            let counted = count.is_some();
            let mut seen = SEEN
                .get_or_init(Mutex::default)
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if seen.insert((*guest_kind, *stride, counted)) {
                tracing::info!(
                    guest_kind,
                    host_kind = 2,
                    stride,
                    first_max = max,
                    counted,
                    "observed BatchCull indirect signature"
                );
            }
        }
    }

    pub(super) fn plan(&self) -> Option<Plan> {
        match *self {
            Self::Direct(g) => Some(Plan::Direct(g)),
            Self::Indirect {
                guest_kind,
                stride,
                max,
                count,
                ..
            } if stride >= 12 && stride.is_multiple_of(4) && max != 0 => {
                let p = Plan::Indirect {
                    guest_kind,
                    stride,
                    max,
                    counted: count.is_some(),
                };
                (p.bytes() <= 1 << 20).then_some(p)
            }
            _ => None,
        }
    }

    pub(super) fn capture(&self, list: &HostList, back: &ID3D12Resource, offset: u64, plan: Plan) {
        let Self::Indirect { args, count, .. } = *self else {
            return;
        };
        let bytes = plan.bytes() - u64::from(count.is_some()) * 4;
        for ((buffer, src), dest, len) in
            std::iter::once((args, offset, bytes)).chain(count.map(|c| (c, offset + bytes, 4)))
        {
            // ExecuteIndirect subsequently reads this exact object at this exact offset.
            // Preserve whichever state the title's bindings left here (possibly UAV).
            let state = list.bound_buffer_state(buffer).unwrap_or(BUFFER_READ);
            list.buffer_state(buffer, BUFFER_READ);
            list.keep(buffer);
            // SAFETY: execute_indirect validated the host argument and count spans and
            // the private destination includes all selected records plus optional count.
            unsafe { list.list.CopyBufferRegion(back, dest, buffer, src, len) };
            list.buffer_state(buffer, state);
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct Coverage {
    pub unique_x: u32,
    pub groups: u64,
    pub count: Option<u32>,
    pub commands: u32,
    pub first_dimensions: Vec<[u32; 3]>,
}

impl Plan {
    pub(super) fn bytes(self) -> u64 {
        match self {
            Self::Direct(_) => 0,
            Self::Indirect {
                stride,
                max,
                counted,
                ..
            } => u64::from(max.saturating_sub(1)) * u64::from(stride) + 12 + 4 * u64::from(counted),
        }
    }
    /// Each command starts GroupID.x at zero. Y/Z and repeated commands
    /// multiply invocations but do not select additional t82 records.
    pub(super) fn coverage(self, data: &[u32]) -> Coverage {
        let mut out = Coverage::default();
        let mut add = |[x, y, z]: [u32; 3]| {
            if out.first_dimensions.len() < 8 {
                out.first_dimensions.push([x, y, z]);
            }
            if x != 0 && y != 0 && z != 0 {
                out.unique_x = out.unique_x.max(x);
                out.groups = out.groups.saturating_add(
                    u64::from(x)
                        .saturating_mul(y.into())
                        .saturating_mul(z.into()),
                );
            }
        };
        match self {
            Self::Direct(g) => {
                add(g);
                out.commands = 1;
            }
            Self::Indirect {
                stride,
                max,
                counted,
                ..
            } => {
                let count = counted.then(|| data[data.len() - 1]);
                let live = count.map_or(max, |n| n.min(max));
                for i in 0..live {
                    let j = i as usize * (stride / 4) as usize;
                    add([data[j], data[j + 1], data[j + 2]]);
                }
                out.count = count;
                out.commands = live;
            }
        }
        out
    }
}
