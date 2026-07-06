//! End-to-end `praxis` smoke: every op composes across a HIP device.
//! Skipped silently if no HIP device is visible.

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "HIP integration tests use compact tensor dimensions, casts, and assertion-first failure paths"
)]

use half::f16;
use hipcore::Device;
use praxis::{CosSinTable, matmul, rms_norm, rope_apply, softmax};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use taxis::{Shape, Tensor};

fn have_gpu() -> bool {
    matches!(hipcore::device::device_count(), Ok(c) if c > 0)
}

fn random_f16(n: usize, seed: u64) -> Vec<f16> {
    let mut r = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| f16::from_f32(r.gen_range(-0.5..0.5)))
        .collect()
}

#[test]
fn praxis_matmul_matches_cpu() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let d = Device::new(0).expect("dev");
    let (m, n, k) = (64, 32, 48);
    let a_host = random_f16(m * k, 1);
    let b_host = random_f16(k * n, 2);
    let a = Tensor::from_host_f16(&d, &a_host, Shape::new(&[m, k])).expect("a");
    let b = Tensor::from_host_f16(&d, &b_host, Shape::new(&[k, n])).expect("b");
    let out = matmul(&a, &b).expect("matmul");
    let got = out.to_host_f16().expect("d2h");
    let expect = kernels::matmul::cpu::matmul_fp16_ref(&a_host, &b_host, m, n, k);
    assert_eq!(got.len(), expect.len());
    for (g, e) in got.iter().zip(expect.iter()) {
        let (gf, ef) = (g.to_f32(), e.to_f32());
        let denom = ef.abs().max(1e-3);
        assert!(
            (gf - ef).abs() / denom < 5e-3,
            "matmul parity diff: got {gf}, expect {ef}"
        );
    }
}

#[test]
fn praxis_rms_norm_runs() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let d = Device::new(0).expect("dev");
    let (m, n) = (4, 256);
    let x_host = random_f16(m * n, 3);
    let w_host = vec![f16::from_f32(1.0); n];
    let x = Tensor::from_host_f16(&d, &x_host, Shape::new(&[m, n])).expect("x");
    let w = Tensor::from_host_f16(&d, &w_host, Shape::new(&[n])).expect("w");
    let y = rms_norm(&x, &w, 1e-5).expect("rms");
    assert_eq!(y.dims(), &[m, n]);
}

#[test]
fn praxis_softmax_runs() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let d = Device::new(0).expect("dev");
    let (m, n) = (2, 256);
    let x_host = random_f16(m * n, 4);
    let x = Tensor::from_host_f16(&d, &x_host, Shape::new(&[m, n])).expect("x");
    let y = softmax(&x).expect("softmax");
    let host = y.to_host_f16().expect("d2h");
    for row in 0..m {
        let s: f32 = host[row * n..(row + 1) * n]
            .iter()
            .map(|v| v.to_f32())
            .sum();
        assert!((s - 1.0).abs() < 5e-2, "row sum {s}");
    }
}

#[test]
fn praxis_rope_apply_runs() {
    if !have_gpu() {
        eprintln!("skipping: no HIP device");
        return;
    }
    let d = Device::new(0).expect("dev");
    let (b, s, h, hd) = (1, 16, 4, 64);
    let qk_host = random_f16(b * s * h * hd, 5);
    let qk = Tensor::from_host_f16(&d, &qk_host, Shape::new(&[b, s, h, hd])).expect("qk");
    let table = CosSinTable::new(s, hd, 10_000.0);
    let out = rope_apply(&qk, &table).expect("rope");
    assert_eq!(out.dims(), &[b, s, h, hd]);
}
