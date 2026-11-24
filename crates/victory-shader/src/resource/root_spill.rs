//! Packed root arguments in older compute pipelines. Match both independent
//! descriptions (RTS0 and FSSM) before resolving a machine load to a binding.
use super::*;
use crate::dxbc::u32_at;
use crate::native::StageIo;

pub(super) fn matches(native: &NativeProgram, signature: &[u8]) -> bool {
    if !matches!(&native.io, StageIo::Cs(cs) if cs.user_sgprs() == 4)
        || native.stage != HwStage::Cs
        || native.layout.is_empty()
    {
        return false;
    }
    let word = |at| u32_at(signature, at).ok();
    let (Some(version @ (1 | 2)), Some(count), Some(table)) = (word(0), word(4), word(8)) else {
        return false;
    };
    if count as usize != native.layout.len() || table < 24 || table % 4 != 0 {
        return false;
    }
    // The FSSM locations start at virtual dword 16 for this variant. Reject
    // tables, holes, split ranges and signatures with other parameter kinds.
    let mut next = 16;
    for slot in 0..count {
        let Some(param) = native.layout.iter().find(|p| u32::from(p.slot) == slot) else {
            return false;
        };
        if u32::from(param.user_data) != next || param.table_offset_dwords != 0 {
            return false;
        }
        let at = table as usize + slot as usize * 12;
        let (Some(kind), Some(data)) = (word(at), word(at + 8)) else {
            return false;
        };
        let data = data as usize;
        if data % 4 != 0 || word(data) != Some(param.base_register) || word(data + 4).is_none() {
            return false;
        }
        let size = match (kind, param.root) {
            (1, RootType::Constants) => word(data + 8),
            (2..=4, RootType::Descriptor) if version == 1 || word(data + 8).is_some() => Some(2),
            _ => None,
        };
        if size != Some(param.sgprs) || param.sgprs == 0 {
            return false;
        }
        let Some(end) = next.checked_add(param.sgprs).filter(|&n| n <= 256) else {
            return false;
        };
        next = end;
    }
    true
}

impl Tracer<'_> {
    pub(super) fn param_value(&self, value: Value) -> Option<(usize, u32)> {
        self.spilled
            .get(&value)
            .copied()
            .or_else(|| self.param_of(self.inputs[value.0 as usize]?))
    }

    pub(super) fn find_root_loads(&mut self) {
        let entry = |reg| {
            self.p
                .inputs
                .iter()
                .find(|i| i.0 == Reg::S(reg))
                .map(|i| i.1)
        };
        let (Some(low), Some(high)) = (entry(0), entry(1)) else {
            return;
        };
        for block in &self.p.blocks {
            for inst in &block.insts {
                let Fields::Smem {
                    base: Operand::Scalar { code, .. },
                    offset,
                    soffset,
                    ..
                } = &inst.fields
                else {
                    continue;
                };
                if !inst.name().starts_with("s_load_dword")
                    || *offset < 0
                    || offset % 4 != 0
                    || self.term(inst, soffset) != Some(Term::Const(0))
                {
                    continue;
                }
                let origin = |reg| inst.read(Reg::S(reg)).map(|v| self.origin(v));
                if origin(*code) != Some(Term::Value(low)) {
                    continue;
                }
                let Some(Term::Value(hi)) = origin(code + 1) else {
                    continue;
                };
                let Some((_, mask)) = self.sop2(hi, &["s_and_b32"]) else {
                    continue;
                };
                if mask != [Term::Value(high), Term::Const(0xff)]
                    && mask != [Term::Const(0xff), Term::Value(high)]
                {
                    continue;
                }
                let mapped: Option<Vec<_>> = inst
                    .writes
                    .iter()
                    .enumerate()
                    .map(|(i, &(_, v))| {
                        let index = 16 + *offset as u32 / 4 + i as u32;
                        self.layout.iter().enumerate().find_map(|(p, rp)| {
                            let dword = index.checked_sub(u32::from(rp.user_data))?;
                            (dword < rp.sgprs).then_some((v, (p, dword)))
                        })
                    })
                    .collect();
                if let Some(words) = mapped {
                    self.spilled.extend(words);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ComputeIo;

    fn shader(words: &[u32]) -> NativeProgram {
        let param = |root, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data,
            slot,
        };
        NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                param(RootType::Constants, 16, 0),
                param(RootType::Descriptor, 18, 1),
            ],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 4 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    fn signature() -> Vec<u8> {
        let words: [u32; 18] = [2, 2, 24, 0, 72, 0, 1, 0, 48, 3, 0, 60, 0, 0, 2, 0, 0, 0];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn track_words(shader: &NativeProgram, rts: &[u8]) -> Result<Resources, ShaderError> {
        let p = crate::ir::build(&shader.code, false, 4)?;
        let container = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: vec![crate::Part {
                fourcc: FourCC::RTS0,
                offset: 0,
                data: rts,
            }],
        };
        super::super::track_shader(&p, shader, &container, None)
    }

    #[test]
    fn packed_load_crosses_from_constants_to_a_complete_root_address() {
        let words = [
            0x8703ff01, 0xff,       // mask the high dword of the incoming address
            0xbe820300, // s2 = incoming low dword
            0xf4080201, 0xfa000000, // s8..11 = table dwords 0..3
            0xf4000005, 0xfa000018, // s0 = *(u32 *)(s[10:11] + 24)
            0xbf810000,
        ];
        let n = shader(&words);
        let rts = signature();
        let res = track_words(&n, &rts).unwrap();
        assert_eq!(
            res.params
                .iter()
                .map(|&(_, p, d)| (p, d))
                .collect::<Vec<_>>(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
        assert_eq!(res.scalar.len(), 1);
        assert_eq!(
            res.scalar[0].1,
            Address {
                param: 1,
                offset: 24,
                dynamic: None
            }
        );

        // Reading the high half from another source cannot produce a root address.
        let mut wrong = words.to_vec();
        wrong.insert(5, 0xbe8b0380); // s11 = 0
        assert!(track_words(&shader(&wrong), &rts).is_err());
    }

    #[test]
    fn emit_binds_spilled_constants_and_a_bounded_root_srv() {
        let words = [
            0x8703ff01, 0xff, 0xbe820300, 0xf4080201,
            0xfa000000, // s8..11 = two constants, SRV address
            0xbe8c030a, 0xbe8d030b, // s12..13 = full SRV address
            0xbe8e03c1, 0xbe8f03ff, 0x11014fac, // raw buffer descriptor
            0xf4200006, 0x10000000, // s_buffer_load_dword s0, s[12:15], s8
            0xbf060900, // compare loaded dword with constant s9
            0xbf810000,
        ];
        let mut n = shader(&words);
        n.used.push(crate::UsedResource {
            kind: crate::ResourceType::Srv,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 18,
            slot: 1,
        });
        let p = crate::ir::build(&n.code, false, 4).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let io = crate::io::raise(&p, &n).unwrap();
        let rts = signature();
        let container = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: vec![crate::Part {
                fourcc: FourCC::RTS0,
                offset: 0,
                data: &rts,
            }],
        };
        let res = super::super::track_shader(&p, &n, &container, Some(&st)).unwrap();
        let mut host =
            crate::emit::emit(&container, &n, &p, &st, &io, &res, crate::emit::Target::ANY)
                .unwrap();
        assert_eq!(host.root_srvs, [1]);
        let ud: Vec<_> = host
            .user_data
            .iter()
            .map(|u| (u.slot, u.dword, u.index, u.root))
            .collect();
        assert!(ud.contains(&(0, 0, 16, RootType::Constants)), "{ud:?}");
        assert!(ud.contains(&(0, 1, 17, RootType::Constants)), "{ud:?}");
        assert!(ud.contains(&(1, 0, 18, RootType::Descriptor)), "{ud:?}");
        use crate::emit::validator::Validator;
        if let Ok(mut validator) = Validator::load(Validator::default_path().as_deref()) {
            validator.validate(&mut host.dxil).unwrap();
        }
    }

    #[test]
    fn requires_matching_hardware_abi_signature_and_bounded_constant_load() {
        let words = [
            0x8703ff01, 0xff, 0xbe820300, 0xf4080201, 0xfa000000, 0xbf810000,
        ];
        let n = shader(&words);
        let rts = signature();
        assert!(track_words(&n, &rts).is_ok());
        let mut other = n.clone();
        other.io = StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [1, 1, 1],
        });
        assert!(track_words(&other, &rts).is_err());
        let mut wrong_size = rts.clone();
        wrong_size[56] = 1;
        assert!(track_words(&n, &wrong_size).is_err());
        assert!(track_words(&n, &rts[..40]).is_err());
        let mut wrong_pointer = words;
        wrong_pointer[1] = 0xf;
        assert!(track_words(&shader(&wrong_pointer), &rts).is_err());
        let mut past_end = words;
        past_end[4] += 4;
        assert!(track_words(&shader(&past_end), &rts).is_err());
    }
}
