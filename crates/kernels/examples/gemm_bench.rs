//! Phase-1 GEMM benchmark — the exit-gate artefact.

#![expect(
    clippy::too_many_arguments,
    reason = "benchmark shape and launch parameters are clearest as explicit arguments"
)]
#![expect(
    clippy::similar_names,
    reason = "m_i32/n_i32/k_i32 (launch args) and m_f64/n_f64/k_f64 (flop-count math) are \
              intentionally parallel per-dimension type conversions, not a naming accident"
)]
//!
//! Sweeps two shapes on the W7900: a small "sanity" shape (256³) to
//! confirm the pipeline, and 4096³ for the headline number. Both use
//! fp16 I/O with fp32 accumulation inside the kernel. Records shape,
//! dtype, measured TFLOPs, and percent of theoretical peak (123 TFLOPs
//! on gfx1100, per dossier `01-rocm-hip-rdna3.md` §3.2).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/target cargo run --release \
//!   -p kernels --example gemm_bench
//! ```

use std::ffi::c_void;
use std::io::Write;
use std::num::TryFromIntError;
use std::time::Instant;
use std::{io, writeln};

use half::f16;
use hipcore::{Device, DeviceBuffer, Event, Stream};
use kernels::matmul::{Variant, launch_matmul_fp16};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const PEAK_TFLOPS: f64 = 123.0;

#[derive(Debug, thiserror::Error)]
enum BenchError {
    #[error(transparent)]
    Hip(#[from] hipcore::Error),
    #[error(transparent)]
    Kernel(#[from] kernels::Error),
    #[error(transparent)]
    Int(#[from] TryFromIntError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

fn random_f16(n: usize, seed: u64) -> Vec<f16> {
    let mut r = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| f16::from_f32(r.gen_range(-0.5..0.5)))
        .collect()
}

fn f16_bytes(v: &[f16]) -> &[u8] {
    // SAFETY: f16 is `#[repr(transparent)] struct f16(u16)`; every
    // two-byte pattern is a valid f16.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2) }
}

#[derive(Debug)]
struct BenchPoint {
    variant: &'static str,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
    elapsed_ms: f64,
    tflops: f64,
    pct_peak: f64,
}

fn bench_one(
    label: &'static str,
    variant: Variant,
    device: &Device,
    m: usize,
    n: usize,
    k: usize,
    warmup: usize,
    iters: usize,
) -> Result<BenchPoint, BenchError> {
    let stream = Stream::new(device)?;
    let a_host = random_f16(m * k, 11);
    let b_host = random_f16(k * n, 22);

    let a_dev = DeviceBuffer::<u8>::from_host(device, f16_bytes(&a_host))?;
    let b_dev = DeviceBuffer::<u8>::from_host(device, f16_bytes(&b_host))?;
    let d_dev = DeviceBuffer::<u8>::alloc(device, m * n * 2)?;

    let m_i32 = i32::try_from(m)?;
    let n_i32 = i32::try_from(n)?;
    let k_i32 = i32::try_from(k)?;

    let launch = |s: &Stream| -> kernels::Result<()> {
        // SAFETY: benchmark allocations match the shapes passed to the launcher.
        unsafe {
            launch_matmul_fp16(
                variant,
                a_dev.as_device_ptr().cast::<c_void>(),
                b_dev.as_device_ptr().cast::<c_void>(),
                d_dev.as_device_ptr().cast::<c_void>(),
                m_i32,
                n_i32,
                k_i32,
                s,
            )
        }
    };

    for _ in 0..warmup {
        launch(&stream)?;
    }
    stream.synchronize()?;

    let start = Event::new(device)?;
    let end = Event::new(device)?;
    stream.record(&start)?;
    for _ in 0..iters {
        launch(&stream)?;
    }
    stream.record(&end)?;
    end.synchronize()?;
    let elapsed_ms = Event::elapsed_ms(&start, &end)?;

    // 2 * M * N * K flops per GEMM.
    let m_f64 = f64::from(u32::try_from(m).unwrap_or(u32::MAX));
    let n_f64 = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let k_f64 = f64::from(u32::try_from(k).unwrap_or(u32::MAX));
    let iters_f64 = f64::from(u32::try_from(iters).unwrap_or(u32::MAX));
    let flops_per_iter = 2.0 * m_f64 * n_f64 * k_f64;
    let total_flops = flops_per_iter * iters_f64;
    let seconds = f64::from(elapsed_ms) / 1000.0;
    let tflops = total_flops / seconds / 1e12;
    let pct_peak = 100.0 * tflops / PEAK_TFLOPS;

    Ok(BenchPoint {
        variant: label,
        m,
        n,
        k,
        iters,
        elapsed_ms: f64::from(elapsed_ms),
        tflops,
        pct_peak,
    })
}

fn main() -> Result<(), BenchError> {
    let mut stdout = io::stdout().lock();
    let device = Device::new(0)?;
    writeln!(
        stdout,
        "Phase-1 GEMM benchmark on {} (isa={})",
        device.props().name,
        device.props().isa
    )?;
    writeln!(stdout, "Peak (datasheet) fp16-WMMA: {PEAK_TFLOPS} TFLOPs")?;
    writeln!(stdout)?;
    writeln!(
        stdout,
        "{:<12} {:>8} {:>8} {:>8} {:>6} {:>12} {:>10} {:>8}",
        "variant", "M", "N", "K", "iters", "elapsed_ms", "TFLOPs", "%peak"
    )?;
    writeln!(stdout, "{:-<78}", "")?;

    // Warm the driver.
    let _t0 = Instant::now();

    let results = [
        bench_one("naive", Variant::Naive, &device, 256, 256, 256, 2, 10)?,
        bench_one("wmma", Variant::Wmma, &device, 256, 256, 256, 2, 50)?,
        bench_one("wmma", Variant::Wmma, &device, 1024, 1024, 1024, 2, 20)?,
        bench_one("wmma", Variant::Wmma, &device, 4096, 4096, 4096, 2, 10)?,
    ];
    for r in &results {
        writeln!(
            stdout,
            "{:<12} {:>8} {:>8} {:>8} {:>6} {:>12.3} {:>10.2} {:>7.1}%",
            r.variant, r.m, r.n, r.k, r.iters, r.elapsed_ms, r.tflops, r.pct_peak
        )?;
    }

    // Emit a single machine-readable record for the headline 4K³ case.
    if let Some(headline) = results
        .iter()
        .find(|r| r.m == 4096 && r.n == 4096 && r.k == 4096)
    {
        writeln!(stdout)?;
        writeln!(stdout, "PHASE1_GEMM_HEADLINE:")?;
        writeln!(
            stdout,
            "  shape: {}x{}x{}",
            headline.m, headline.n, headline.k
        )?;
        writeln!(stdout, "  dtype: fp16_in_fp32_acc_fp16_out")?;
        writeln!(stdout, "  variant: {}", headline.variant)?;
        writeln!(stdout, "  iters: {}", headline.iters)?;
        writeln!(stdout, "  elapsed_ms: {:.3}", headline.elapsed_ms)?;
        writeln!(stdout, "  tflops: {:.2}", headline.tflops)?;
        writeln!(stdout, "  pct_peak: {:.2}", headline.pct_peak)?;
    }
    Ok(())
}
