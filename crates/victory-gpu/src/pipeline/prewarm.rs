//! Prepare the exact device PSOs of a captured compute workload before the guest starts.
use super::{Shader, host::HostPipeline};
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use victory_shader::{Container, FourCC, HwStage, native_programs};

// Shaders are interned by full content hash. Each cached pipeline retains its
// shader, so the address cannot be reused during the life of the cache.
static BUILT: OnceLock<HashMap<usize, HostPipeline>> = OnceLock::new();

pub(super) fn lookup(cs: &Arc<Shader>) -> Option<&'static HostPipeline> {
    BUILT.get()?.get(&(Arc::as_ptr(cs) as usize))
}

pub fn prewarm_compute_directory(dir: &Path) -> io::Result<()> {
    let mut files = Vec::new();
    for entry in dir.read_dir()? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|ext| ext == "compute")
            && entry.file_type()?.is_file()
        {
            files.push(entry.path());
        }
    }
    files.sort_unstable();
    let started = std::time::Instant::now();
    let count = files.len();
    if count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no .compute shader captures to prewarm",
        ));
    }
    tracing::info!(count, "preparing compute pipelines before guest startup");
    let built = std::thread::scope(|s| -> io::Result<HashMap<usize, HostPipeline>> {
        let mut workers = Vec::new();
        for chunk in files.chunks(count.div_ceil(2)) {
            workers.push(s.spawn(move || -> io::Result<Vec<(usize, HostPipeline)>> {
                let mut built = Vec::new();
                for file in chunk {
                    let bytes = std::fs::read(file)?;
                    let native_cs = Container::parse(&bytes)
                        .ok()
                        .and_then(|c| native_programs(c.part(FourCC::DXIL)?.data).ok())
                        .is_some_and(|programs| {
                            !programs.is_empty() && programs.iter().all(|p| p.stage == HwStage::Cs)
                        });
                    if !native_cs {
                        tracing::warn!(?file, "capture lacks native compute code");
                        continue;
                    }
                    let cs = super::shader("compute", &bytes);
                    if let Some(pso) = HostPipeline::new(std::slice::from_ref(&cs), None) {
                        built.push((Arc::as_ptr(&cs) as usize, pso));
                    }
                }
                Ok(built)
            }));
        }
        let mut all = HashMap::new();
        for worker in workers {
            all.extend(
                worker
                    .join()
                    .map_err(|_| io::Error::other("compute prewarm worker panicked"))??,
            );
        }
        Ok(all)
    })?;
    let ready = built.len();
    BUILT
        .set(built)
        .map_err(|_| io::Error::other("compute prewarm already finished"))?;
    super::save_pipeline_cache();
    tracing::info!(
        count,
        ready,
        seconds = started.elapsed().as_secs_f64(),
        "compute prewarm complete"
    );
    Ok(())
}
