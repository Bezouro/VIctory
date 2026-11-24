//! Run DXC's CFG passes on an already translated compute shader. DXC serializes
//! the optimized module and its signatures back into an ordinary shader container.
use super::{BLOB_VTABLE, Blob, CreateInstance, Guid, Object, method, release};
use std::ptr::{NonNull, null_mut};

const OPTIMIZER: Guid = Guid(
    0xae2cd79f,
    0xcc22,
    0x453f,
    [0x9b, 0x6b, 0xb1, 0x24, 0xe7, 0xa5, 0x20, 0x4c],
);
const IOPTIMIZER: Guid = Guid(
    0x25740e2e,
    0x9cba,
    0x401b,
    [0x91, 0x19, 0x4f, 0xb4, 0x2f, 0x39, 0xf2, 0x70],
);
const ASSEMBLER: Guid = Guid(
    0xd728db68,
    0xf903,
    0x4f80,
    [0x94, 0xcd, 0xdc, 0xcf, 0x76, 0xec, 0x71, 0x51],
);
const IASSEMBLER: Guid = Guid(
    0x091f7a26,
    0x1c1f,
    0x4948,
    [0x90, 0x4b, 0xe6, 0xe3, 0xa8, 0xa7, 0x71, 0xd5],
);

struct Owned(Object);
impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: a successful COM call transferred one reference.
        unsafe { release(self.0) };
    }
}

fn output(op: &str, call: impl FnOnce(*mut Object) -> i32) -> Result<Owned, String> {
    let mut out = null_mut();
    let hr = call(&mut out);
    let object = NonNull::new(out).map(|p| Owned(p.as_ptr()));
    object
        .filter(|_| hr >= 0)
        .ok_or_else(|| format!("DXC {op} failed ({hr:#x})"))
}

pub(crate) fn compute(mut dxil: Vec<u8>) -> Result<Vec<u8>, String> {
    let library = crate::hostlib::dxcompiler()?;
    let address = library.symbol(c"DxcCreateInstance")?;
    let mut blob = Blob {
        vtable: &BLOB_VTABLE,
        data: dxil.as_mut_ptr(),
        len: dxil.len(),
    };
    let names = ["-simplifycfg", "-loop-simplify", "-lcssa"]
        .map(|name| name.chars().map(u32::from).chain([0]).collect::<Vec<_>>());
    let passes = names.each_ref().map(|name| name.as_ptr());
    // SAFETY: these are the public DXC COM interfaces. Linux DXC uses 32-bit
    // wchar_t and SysV. Input buffers and pass strings live through both calls.
    unsafe {
        let create: CreateInstance = std::mem::transmute(address);
        let opt = output("create optimizer", |p| {
            create(&OPTIMIZER, &IOPTIMIZER, p.cast())
        })?;
        let run: unsafe extern "system" fn(
            Object,
            *mut Blob,
            *const *const u32,
            u32,
            *mut Object,
            *mut Object,
        ) -> i32 = method(opt.0, 5);
        let module = output("optimize", |p| {
            run(
                opt.0,
                &mut blob,
                passes.as_ptr(),
                passes.len() as u32,
                p,
                null_mut(),
            )
        })?;
        let asm = output("create assembler", |p| {
            create(&ASSEMBLER, &IASSEMBLER, p.cast())
        })?;
        let assemble: unsafe extern "system" fn(Object, Object, *mut Object) -> i32 =
            method(asm.0, 3);
        let result = output("assemble", |p| assemble(asm.0, module.0, p))?;
        let get_status: unsafe extern "system" fn(Object, *mut i32) -> i32 = method(result.0, 3);
        let mut status = -1;
        let hr = get_status(result.0, &mut status);
        if hr < 0 || status < 0 {
            return Err(format!("DXC assembly failed ({hr:#x}, {status:#x})"));
        }
        let get_result: unsafe extern "system" fn(Object, *mut Object) -> i32 = method(result.0, 4);
        let out = output("get assembled shader", |p| get_result(result.0, p))?;
        let data: unsafe extern "system" fn(Object) -> *const u8 = method(out.0, 3);
        let size: unsafe extern "system" fn(Object) -> usize = method(out.0, 4);
        let (data, size) = (data(out.0), size(out.0));
        if data.is_null() || size < 32 {
            return Err("DXC returned an empty shader".into());
        }
        Ok(std::slice::from_raw_parts(data, size).to_vec())
    }
}
