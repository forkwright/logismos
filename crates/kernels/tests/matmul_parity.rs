//! HIP-backed matmul parity tests against the CPU reference.
//!
//! Skipped silently if no HIP device is visible.

#![expect(
    clippy::expect_used,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "GPU parity tests use compact math notation, host/device casts, and assertion-first failure paths"
)]

use std::ffi::c_void;

use half::f16;
use hipcore::{Device, DeviceBuffer, Stream};
use kernels::matmul::{Variant, cpu, launch_matmul_fp16};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn have_gpu() -> bool {
    matches!(hipcore::device::device_count(), Ok(c) if c > 0)
}

fn run_gpu_matmul(
    variant: Variant,
    a: &[f16],
    b: &[f16],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f16> {
    let device = Device::new(0).expect("open dev 0");
    let stream = Stream::new(&device).expect("stream");

    let a_bytes = unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), a.len() * 2) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u8>(), b.len() * 2) };

    let a_dev = DeviceBuffer::<u8>::from_host(&device, a_bytes).expect("alloc a");
    let b_dev = DeviceBuffer::<u8>::from_host(&device, b_bytes).expect("alloc b");
    let d_dev = DeviceBuffer::<u8>::alloc(&device, m * n * 2).expect("alloc d");

    // SAFETY: allocations match shapes; pointers live until `synchronize`.
    unsafe {
        launch_matmul_fp16(
            variant,
            a_dev.as_device_ptr().cast::<c_void>(),
            b_dev.as_device_ptr().cast::<c_void>(),
            d_dev.as_device_ptr().cast::<c_void>(),
            m as i32,
            n as i32,
            k as i32,
            &stream,
        )
        .expect("launch");
    }
    stream.synchronize().expect("sync");

    let mut out_bytes = vec![0u8; m * n * 2];
    d_dev.copy_to_host(&mut out_bytes).expect("d2h");
    // SAFETY: `f16` is `#[repr(transparent)] struct f16(u16)`; every
    // 2-byte pattern is a valid inhabitant.
    let out = unsafe { std::slice::from_raw_parts(out_bytes.as_ptr().cast::<f16>(), m * n) };
    out.to_vec()
}

fn random_f16_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f16> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| f16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect()
}

fn parity_check(gpu: &[f16], cpu: &[f16], tol: f32) {
    assert_eq!(gpu.len(), cpu.len());
    let mut max_rel: f32 = 0.0;
    let mut max_abs: f32 = 0.0;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let gf = g.to_f32();
        let cf = c.to_f32();
        let abs_err = (gf - cf).abs();
        max_abs = max_abs.max(abs_err);
        let denom = cf.abs().max(1e-3);
        max_rel = max_rel.max(abs_err / denom);
    }
    assert!(
        max_rel <= tol,
        "parity failed: max_rel={max_rel:.5} max_abs={max_abs:.5} (tol={tol})"
    );
}

#[test]
fn matmul_naive_parity_small() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let (m, n, k) = (32, 48, 64);
    let a = random_f16_matrix(m, k, 1);
    let b = random_f16_matrix(k, n, 2);
    let cpu_out = cpu::matmul_fp16_ref(&a, &b, m, n, k);
    let gpu_out = run_gpu_matmul(Variant::Naive, &a, &b, m, n, k);
    parity_check(&gpu_out, &cpu_out, 5e-3);
}

#[test]
fn matmul_wmma_parity_small() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    // All dims multiple of 16 — WMMA tile.
    let (m, n, k) = (64, 64, 64);
    let a = random_f16_matrix(m, k, 3);
    let b = random_f16_matrix(k, n, 4);
    let cpu_out = cpu::matmul_fp16_ref(&a, &b, m, n, k);
    let gpu_out = run_gpu_matmul(Variant::Wmma, &a, &b, m, n, k);
    parity_check(&gpu_out, &cpu_out, 5e-3);
}

#[test]
fn matmul_wmma_parity_medium() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let (m, n, k) = (256, 256, 256);
    let a = random_f16_matrix(m, k, 5);
    let b = random_f16_matrix(k, n, 6);
    let cpu_out = cpu::matmul_fp16_ref(&a, &b, m, n, k);
    let gpu_out = run_gpu_matmul(Variant::Wmma, &a, &b, m, n, k);
    parity_check(&gpu_out, &cpu_out, 5e-3);
}
