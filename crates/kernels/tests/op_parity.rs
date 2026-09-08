//! HIP-backed parity tests for the non-matmul tier-1 kernels.

#![expect(
    clippy::expect_used,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "GPU parity tests use compact math notation, host/device casts, and assertion-first failure paths"
)]

use std::ffi::c_void;

use half::f16;
use hipcore::{Device, DeviceBuffer, Stream};
use kernels::rms_norm::{cpu as rms_cpu, launch_rms_norm_fp16};
use kernels::rope::{cpu as rope_cpu, launch_rope_fp16_in_place};
use kernels::softmax::launch_softmax_fp16;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn have_gpu() -> bool {
    matches!(hipcore::device::device_count(), Ok(c) if c > 0)
}

fn parity_check(gpu: &[f16], cpu: &[f16], tol: f32) {
    assert_eq!(gpu.len(), cpu.len());
    let mut max_rel: f32 = 0.0;
    let mut max_abs: f32 = 0.0;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let gf = g.to_f32();
        let cf = c.to_f32();
        let abs = (gf - cf).abs();
        max_abs = max_abs.max(abs);
        let denom = cf.abs().max(1e-3);
        max_rel = max_rel.max(abs / denom);
    }
    assert!(
        max_rel <= tol,
        "parity failed: max_rel={max_rel:.5} max_abs={max_abs:.5} (tol={tol})"
    );
}

fn random_f16(n: usize, seed: u64) -> Vec<f16> {
    let mut r = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| f16::from_f32(r.random_range(-1.0..1.0)))
        .collect()
}

fn f16_bytes(v: &[f16]) -> &[u8] {
    // SAFETY: f16 is `#[repr(transparent)] struct f16(u16)`, and
    // every two-byte pattern is a valid f16.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 2) }
}

fn bytes_to_f16(v: &[u8]) -> Vec<f16> {
    assert_eq!(v.len() % 2, 0);
    // WHY not a `*const f16` reinterpret: `from_raw_parts` requires the pointer
    // to be aligned for `f16`, and a `&[u8]` guarantees only 1-byte alignment.
    // The old SAFETY comment established bit-pattern totality but never that
    // precondition. Chunking is alignment-independent and `from_ne_bytes` keeps
    // the native byte order the reinterpret read.
    v.chunks_exact(2)
        .map(|pair| {
            let mut bytes = [0u8; 2];
            bytes.copy_from_slice(pair);
            f16::from_ne_bytes(bytes)
        })
        .collect()
}

/// Independent f64 softmax reference for the bounded finite inputs in the
/// hardware-parity witness.
///
/// This deliberately uses the direct exponent expression rather than the
/// production max-subtraction implementation. It is not a production masking
/// policy: an all-negative-infinity row remains undefined here, so parity only
/// supplies `[-1, 1]` finite logits from [`random_f16`].
fn finite_softmax_fp16_oracle(x: &[f16], rows: usize, width: usize) -> Vec<f16> {
    assert!(width > 0, "oracle requires a non-empty last axis");
    let expected_len = rows
        .checked_mul(width)
        .expect("test-only oracle shape must fit usize");
    assert_eq!(x.len(), expected_len, "oracle input shape");
    assert!(
        x.iter().all(|value| value.to_f32().is_finite()),
        "oracle is only defined for finite parity logits"
    );

    let mut output = Vec::with_capacity(expected_len);
    for row in x.chunks_exact(width) {
        let denominator = row
            .iter()
            .map(|value| f64::from(value.to_f32()).exp())
            .sum::<f64>();
        output
            .extend(row.iter().map(|value| {
                f16::from_f32((f64::from(value.to_f32()).exp() / denominator) as f32)
            }));
    }
    output
}

#[test]
fn finite_softmax_oracle_matches_known_ratios() {
    let logits = [
        f16::from_f32(0.0),
        f16::from_f32(std::f32::consts::LN_2),
        f16::from_f32(2.0 * std::f32::consts::LN_2),
    ];
    let actual = finite_softmax_fp16_oracle(&logits, 1, 3);
    let expected = [1.0_f32 / 7.0, 2.0 / 7.0, 4.0 / 7.0];
    for (value, expected) in actual.iter().zip(expected) {
        assert!(
            (value.to_f32() - expected).abs() < 1e-3,
            "got {value:?}, expected {expected}"
        );
    }
}

#[test]
fn rms_norm_parity() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let device = Device::new(0).expect("open");
    let stream = Stream::new(&device).expect("stream");

    let m = 8;
    let n = 1024;
    let eps = 1e-5;
    let x = random_f16(m * n, 42);
    let w = random_f16(n, 43);

    let cpu = rms_cpu::rms_norm_fp16_ref(&x, &w, m, n, eps);

    let x_dev = DeviceBuffer::<u8>::from_host(&device, f16_bytes(&x)).expect("x");
    let w_dev = DeviceBuffer::<u8>::from_host(&device, f16_bytes(&w)).expect("w");
    let y_dev = DeviceBuffer::<u8>::alloc(&device, m * n * 2).expect("y");
    // SAFETY: pointers valid; sizes match above.
    unsafe {
        launch_rms_norm_fp16(
            x_dev.as_device_ptr().cast::<c_void>(),
            w_dev.as_device_ptr().cast::<c_void>(),
            y_dev.as_device_ptr().cast::<c_void>(),
            m as i32,
            n as i32,
            eps,
            &stream,
        )
        .expect("launch rms");
    }
    stream.synchronize().expect("sync");
    let mut out_bytes = vec![0u8; m * n * 2];
    y_dev.copy_to_host(&mut out_bytes).expect("d2h");
    let gpu = bytes_to_f16(&out_bytes);

    parity_check(&gpu, &cpu, 5e-3);
}

#[test]
fn softmax_parity() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let device = Device::new(0).expect("open");
    let stream = Stream::new(&device).expect("stream");

    let m = 4;
    let n = 512;
    let x = random_f16(m * n, 7);
    let cpu = finite_softmax_fp16_oracle(&x, m, n);

    let x_dev = DeviceBuffer::<u8>::from_host(&device, f16_bytes(&x)).expect("x");
    let y_dev = DeviceBuffer::<u8>::alloc(&device, m * n * 2).expect("y");
    // SAFETY: pointers valid; sizes match above.
    unsafe {
        launch_softmax_fp16(
            x_dev.as_device_ptr().cast::<c_void>(),
            y_dev.as_device_ptr().cast::<c_void>(),
            m as i32,
            n as i32,
            &stream,
        )
        .expect("launch softmax");
    }
    stream.synchronize().expect("sync");
    let mut out_bytes = vec![0u8; m * n * 2];
    y_dev.copy_to_host(&mut out_bytes).expect("d2h");
    let gpu = bytes_to_f16(&out_bytes);

    parity_check(&gpu, &cpu, 5e-3);
}

#[test]
fn rope_parity() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let device = Device::new(0).expect("open");
    let stream = Stream::new(&device).expect("stream");

    let batch = 2;
    let seq = 64;
    let heads = 8;
    let head_dim = 128;
    let theta = 10_000.0f32;

    let qk_host = random_f16(batch * seq * heads * head_dim, 13);
    let cos_sin = rope_cpu::build_cos_sin_table(seq, head_dim, theta);

    // CPU reference, in-place on a mutable clone.
    let mut cpu = qk_host.clone();
    rope_cpu::rope_apply_fp16_ref(&mut cpu, &cos_sin, batch, seq, heads, head_dim)
        .expect("shapes match by construction");

    // GPU launch, in-place on a device copy.
    let qk_dev = DeviceBuffer::<u8>::from_host(&device, f16_bytes(&qk_host)).expect("qk");
    // Reinterpret cos_sin as bytes for the device upload.
    let cs_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(cos_sin.as_ptr().cast::<u8>(), cos_sin.len() * 4) };
    let cs_dev = DeviceBuffer::<u8>::from_host(&device, cs_bytes).expect("cos_sin");

    // SAFETY: pointers valid; sizes match above.
    unsafe {
        launch_rope_fp16_in_place(
            qk_dev.as_device_ptr().cast::<c_void>(),
            cs_dev.as_device_ptr().cast::<c_void>(),
            batch as i32,
            seq as i32,
            heads as i32,
            head_dim as i32,
            &stream,
        )
        .expect("launch rope");
    }
    stream.synchronize().expect("sync");

    let mut out_bytes = vec![0u8; qk_host.len() * 2];
    qk_dev.copy_to_host(&mut out_bytes).expect("d2h");
    let gpu = bytes_to_f16(&out_bytes);

    parity_check(&gpu, &cpu, 5e-3);
}
