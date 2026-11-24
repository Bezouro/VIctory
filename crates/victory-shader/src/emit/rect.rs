//! Geometry stage for screen-aligned rectangles produced by vertex number.
//! Adapted from Gargalindis' VIctory #1020. The corner opposite the diagonal
//! is the shared corner. Each tile's three original invocations keep their IDs.
use super::{container::SigElement, signature};
use crate::{Container, FourCC};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{Arc, LazyLock, Mutex};

fn elements(c: &Container<'_>, part: FourCC) -> Result<Vec<SigElement>, String> {
    signature(c.part(part).ok_or("missing stage signature")?.data).map_err(|e| e.to_string())
}

fn word(d: &[u8], at: usize) -> Option<usize> {
    Some(u32::from_le_bytes(d.get(at..at + 4)?.try_into().ok()?) as usize)
}

/// Locate input element records in PSV0 (the interpolator mode is absent
/// from ISG1). Rows can hold multiple fields, down to single components.
fn interpolation(d: &[u8]) -> Option<HashMap<(u32, u8), u8>> {
    let info = word(d, 0)?;
    if info < 32 || *d.get(4 + 24)? != 0 {
        return None;
    }
    let count = usize::from(*d.get(4 + 28)?);
    let mut at = 4 + info;
    let bindings = word(d, at)?;
    at += 4;
    if bindings != 0 {
        at = at.checked_add(4 + bindings.checked_mul(word(d, at)?)?)?;
    }
    at = at.checked_add(4 + word(d, at)?)?; // string table, length in bytes
    at = at.checked_add(4 + 4 * word(d, at)?)?; // semantic indices, in words
    let size = word(d, at)?;
    if size < 16 {
        return None;
    }
    at += 4;
    let mut modes = HashMap::new();
    for i in 0..count {
        let e = d.get(at + i * size..at + i * size + 16)?;
        if e[8] != 1 {
            return None;
        }
        if e[10] & 0x40 != 0 {
            modes.insert((u32::from(e[9]), (e[10] >> 4) & 3), e[13]);
        }
    }
    Some(modes)
}

fn source(vs: &[u8], ps: &[u8]) -> Result<(String, Vec<SigElement>), String> {
    let vs = Container::parse(vs).map_err(|e| e.to_string())?;
    let ps = Container::parse(ps).map_err(|e| e.to_string())?;
    // GetAttributeAtVertex can observe different values even for a flat
    // field, including when the PS has no SV_Barycentrics signature entry.
    let features = ps
        .part(FourCC::SFI0)
        .and_then(|p| p.data.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or("missing pixel feature flags")?;
    if features & 0x2_0000 != 0 {
        return Err("rectangle pixel shader observes the original triangle vertices".into());
    }
    let inputs = elements(&vs, FourCC::ISG1)?;
    if !inputs.iter().any(|e| e.kind == 1) || inputs.iter().any(|e| !matches!(e.kind, 1 | 2)) {
        return Err("rectangle vertex shader requires a vertex buffer or lacks SV_VertexID".into());
    }
    let pixels = elements(&ps, FourCC::ISG1)?;
    if pixels.iter().any(|e| !matches!(e.kind, 0 | 3)) {
        return Err("rectangle pixel shader reads an unsupported system value".into());
    }
    let modes = interpolation(ps.part(FourCC::PSV0).ok_or("missing PSV0")?.data)
        .ok_or("unsupported pixel interpolation layout")?;
    let outputs = elements(&vs, FourCC::OSG1)?;
    let mut fields = String::new();
    let mut fourth = String::new();
    let mut flat = String::new();
    let mut position = None;
    for (n, e) in outputs.iter().enumerate() {
        if e.kind == 3 && e.cols == 4 && e.comp == 9 {
            position = Some(n)
        } else if e.kind != 0 {
            return Err("unsupported rectangle vertex system output".into());
        }
        if !e
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || e.half
        {
            return Err("unsupported rectangle semantic or precision".into());
        }
        let ty = match e.comp {
            4 => "int",
            5 => "uint",
            9 => "float",
            _ => return Err("unsupported rectangle component".into()),
        };
        let pix = pixels
            .iter()
            .find(|p| p.index == e.index && p.name.eq_ignore_ascii_case(&e.name));
        let flat_field = match pix {
            Some(p) if p.kind == 0 => {
                let mode = *modes
                    .get(&(p.row.ok_or("unassigned pixel input")?, p.col))
                    .ok_or("missing pixel interpolation")?;
                match mode {
                    1 => true,
                    2..=7 if e.comp == 9 => false,
                    _ => return Err("unsupported pixel interpolation".into()),
                }
            }
            _ => e.comp != 9,
        };
        let semantic = if e.kind == 3 {
            "SV_Position".to_string()
        } else {
            format!("{}{}", e.name, e.index)
        };
        let qualifier = if flat_field { "nointerpolation " } else { "" };
        writeln!(fields, "  {qualifier}{ty}{} e{n} : {semantic};", e.cols).unwrap();
        if flat_field {
            writeln!(flat, "  v0.e{n} = v1.e{n} = v2.e{n} = d.e{n} = i[0].e{n};").unwrap();
        } else {
            writeln!(fourth, "  d.e{n} = i[s].e{n} + (i[t].e{n} - i[r].e{n});").unwrap();
        }
    }
    let p = position.ok_or("rectangle has no float4 SV_Position")?;
    let hlsl = format!(
        r#"struct V {{
{fields}}};
[maxvertexcount(4)]
void main(triangle V i[3], inout TriangleStream<V> stream) {{
  float4 a = i[0].e{p}, b = i[1].e{p}, c = i[2].e{p};
  uint r = 3;
  if ((a.x == b.x && a.y == c.y) || (a.x == c.x && a.y == b.y)) r = 0;
  else if ((b.x == a.x && b.y == c.y) || (b.x == c.x && b.y == a.y)) r = 1;
  else if ((c.x == a.x && c.y == b.y) || (c.x == b.x && c.y == a.y)) r = 2;
  if (r == 3 || a.w != b.w || a.w != c.w) {{
    stream.Append(i[0]); stream.Append(i[1]); stream.Append(i[2]); return;
  }}
  uint s = (r+1)%3, t = (r+2)%3;
  V v0 = i[r], v1 = i[s], v2 = i[t], d;
{fourth}{flat}  d.e{p}.x = i[s].e{p}.x == i[r].e{p}.x ? i[t].e{p}.x : i[s].e{p}.x;
  d.e{p}.y = i[s].e{p}.y == i[r].e{p}.y ? i[t].e{p}.y : i[s].e{p}.y;
  d.e{p}.w = i[r].e{p}.w;
  stream.Append(v0); stream.Append(v1); stream.Append(v2); stream.Append(d);
}}"#
    );
    Ok((hlsl, outputs))
}

/// Generate and cache a GS compatible with these translated vertex and
/// pixel stages. Check both sides of the GS before linking into the PSO.
pub fn procedural_rectangle(vs: &[u8], ps: &[u8]) -> Result<Arc<[u8]>, String> {
    type Cache = Mutex<HashMap<String, Result<Arc<[u8]>, String>>>;
    static CACHE: LazyLock<Cache> = LazyLock::new(Mutex::default);
    let (source, original) = source(vs, ps)?;
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let stage = cache
        .entry(source)
        .or_insert_with_key(|s| crate::hostlib::compile_hlsl(s, "main", "gs_6_0").map(Arc::from))
        .clone()?;
    let c = Container::parse(&stage).map_err(|e| e.to_string())?;
    for part in [FourCC::ISG1, FourCC::OSG1] {
        let got = elements(&c, part)?;
        if original.len() != got.len()
            || !original.iter().all(|e| {
                got.iter().any(|g| {
                    e.name.eq_ignore_ascii_case(&g.name)
                        && (e.index, e.row, e.col, e.cols, e.comp, e.kind)
                            == (g.index, g.row, g.col, g.cols, g.comp, g.kind)
                })
            })
        {
            return Err(format!(
                "rectangle geometry {part} does not match the vertex layout"
            ));
        }
    }
    Ok(stage)
}
