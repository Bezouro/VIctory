//! Fingerprint the inputs to shader conversion independently of the window,
//! guest loader and GPU submission code.
use std::{env, fs, io, path::Path};

fn hash_path(hash: &mut blake3::Hasher, path: &Path) -> io::Result<()> {
    if path.is_dir() {
        let mut entries = fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<io::Result<Vec<_>>>()?;
        entries.sort();
        for entry in entries {
            hash_path(hash, &entry)?;
        }
    } else {
        let name = path.to_string_lossy();
        hash.update(&(name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(blake3::hash(&fs::read(path)?).as_bytes());
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let mut hash = blake3::Hasher::new();
    for input in [
        "../victory-shader/src",
        "../victory-shader/Cargo.toml",
        "src/shader_cache.rs",
        "build.rs",
        "../../Cargo.toml",
        "../../Cargo.lock",
    ] {
        println!("cargo::rerun-if-changed={input}");
        hash_path(&mut hash, Path::new(input))?;
    }
    for key in ["TARGET", "CARGO_CFG_TARGET_FEATURE", "CARGO_ENCODED_RUSTFLAGS"] {
        let value = env::var(key).unwrap_or_default();
        hash.update(&(value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    let output = Path::new(&env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"))
        .join("shader-fingerprint.bin");
    fs::write(output, hash.finalize().as_bytes())
}
