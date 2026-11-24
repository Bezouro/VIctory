/// SSE4a's `extrq`/`insertq` (register forms, the only ones there are) on
/// the saved XMM registers; returns the instruction's length.
pub(crate) fn emulate(byte: impl Fn(usize) -> Option<u8>, xmm: &mut [[u64; 2]; 16]) -> Option<usize> {
    let prefix = byte(0)?;
    let (rex, at) = if byte(1)? & 0xf0 == 0x40 {
        (byte(1)?, 2)
    } else {
        (0, 1)
    };
    if !matches!(prefix, 0x66 | 0xf2) || byte(at)? != 0x0f || !matches!(byte(at + 1)?, 0x78 | 0x79) {
        return None;
    }
    let modrm = byte(at + 2)?;
    if modrm >> 6 != 3 {
        return None;
    }
    let dst = usize::from((modrm >> 3) & 7 | (rex & 4) << 1);
    let src = usize::from(modrm & 7 | (rex & 1) << 3);
    let immediate = byte(at + 1)? == 0x78;
    let (fields, len) = if immediate {
        ((byte(at + 3)?, byte(at + 4)?), at + 5)
    } else {
        // extrq reads its fields from src's low half, insertq from its high.
        let f = if prefix == 0x66 { xmm[src][0] } else { xmm[src][1] };
        ((f as u8, (f >> 8) as u8), at + 3)
    };
    let (n, at_bit) = (u32::from(fields.0 & 63), u32::from(fields.1 & 63));
    let bits = if n == 0 { 64 } else { n };
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    if prefix == 0x66 {
        // extrq: register form takes its fields from src, works on dst.
        if immediate && (modrm >> 3) & 7 != 0 {
            return None;
        }
        let target = if immediate { src } else { dst };
        xmm[target][0] = (xmm[target][0] >> at_bit) & mask;
    } else {
        let value = (xmm[src][0] & mask) << at_bit;
        xmm[dst][0] = xmm[dst][0] & !(mask << at_bit) | value;
    }
    Some(len)
}
