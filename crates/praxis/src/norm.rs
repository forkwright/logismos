//! `praxis::rms_norm` — row-wise RMSNorm.

use std::ffi::c_void;

use taxis::{DType, Tensor};

use crate::device_dispatch::{DevicePlacement, classify_placement};
use crate::error::{InvalidSnafu, Result};

fn dim_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        InvalidSnafu {
            op: "rms_norm",
            msg: format!("{name} dimension exceeds i32: {value}"),
        }
        .build()
    })
}

/// `y = x / sqrt(mean(x .* x) + eps) .* weight`, per row.
///
/// Phase 1 contract: `x` is 2-D `(M, N)`, fp16, contiguous; `weight`
/// is 1-D `(N,)`, fp16; `eps` is f32.
///
/// # Errors
///
/// See [`Error::Invalid`] and propagated kernel errors.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if x.dtype() != DType::F16 || weight.dtype() != DType::F16 {
        return InvalidSnafu {
            op: "rms_norm",
            msg: "F16 only in Phase 1".into(),
        }
        .fail();
    }
    if x.dims().len() != 2 {
        return InvalidSnafu {
            op: "rms_norm",
            msg: format!("expected (M, N); got {:?}", x.dims()),
        }
        .fail();
    }
    let x_dims = x.dims();
    let (Some(&m), Some(&n)) = (x_dims.first(), x_dims.get(1)) else {
        return InvalidSnafu {
            op: "rms_norm",
            msg: "input rank changed during validation".into(),
        }
        .fail();
    };
    let weight_dims = weight.dims();
    if weight_dims.len() != 1 || weight_dims.first().copied() != Some(n) {
        return InvalidSnafu {
            op: "rms_norm",
            msg: format!("weight shape {:?} != [{}]", weight.dims(), n),
        }
        .fail();
    }

    let placement = classify_placement(x.hip_storage().is_some(), weight.hip_storage().is_some());
    match placement {
        DevicePlacement::BothHip => {
            let (Some(x_hip), Some(w_hip)) = (x.hip_storage(), weight.hip_storage()) else {
                return InvalidSnafu {
                    op: "rms_norm",
                    msg: "device-pair classification invariant violated: BothHip without two HIP operands".into(),
                }.fail();
            };
            let device = x_hip.device();
            let out = Tensor::zeros_hip(device, DType::F16, x.shape().clone())?;
            let out_hip = out.hip_storage().ok_or_else(|| {
                InvalidSnafu {
                    op: "rms_norm",
                    msg: "zeros_hip did not return HIP".into(),
                }
                .build()
            })?;
            crate::stream_pool::POOL.with_stream(device, |stream| {
                // SAFETY: device pointers valid; sizes verified above.
                // `out` was just allocated above and not yet shared with
                // any other `Tensor` handle, so `out_hip`'s
                // `as_mut_device_ptr` obligation (no other live pointer to
                // this allocation) holds by construction.
                unsafe {
                    kernels::rms_norm::launch_rms_norm_fp16(
                        x_hip.as_device_ptr().cast::<c_void>(),
                        w_hip.as_device_ptr().cast::<c_void>(),
                        out_hip.as_mut_device_ptr().cast::<c_void>(),
                        dim_i32(m, "M")?,
                        dim_i32(n, "N")?,
                        eps,
                        stream,
                    )?;
                }
                stream.synchronize()?;
                Ok(())
            })?;
            Ok(out)
        }
        DevicePlacement::BothCpu => {
            let x_host = x.to_host_f16()?;
            let w_host = weight.to_host_f16()?;
            let y = kernels::rms_norm::cpu::rms_norm_fp16_ref(&x_host, &w_host, m, n, eps);
            Ok(Tensor::from_cpu(
                taxis::CpuStorage::F16(y),
                x.shape().clone(),
            ))
        }
        // WHY(forkwright/logismos#38): a mixed pair used to fall through
        // the old wildcard arm into the CPU path above, silently
        // transferring the HIP operand to host with no error and no
        // diagnostic. `x` and `weight` disagreeing on device is a
        // caller bug (typically a weight upload that never reached the
        // device) and must fail loudly (CLAUDE.md:77, AGENTS.md:29 — no
        // silent CPU fallbacks).
        DevicePlacement::Mixed => InvalidSnafu {
            op: "rms_norm",
            msg: "x and weight must be on the same device".into(),
        }
        .fail(),
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test assertions use expect()/expect_err() directly"
    )]

    use half::f16;
    use taxis::{CpuStorage, Shape, Tensor};

    use super::rms_norm;

    fn f16_tensor(values: &[f32], shape: &[usize]) -> Tensor {
        let data: Vec<f16> = values.iter().copied().map(f16::from_f32).collect();
        Tensor::from_cpu(CpuStorage::F16(data), Shape::new(shape))
    }

    /// WHY: exercises the `DevicePlacement::BothCpu` arm through the
    /// real public `rms_norm` entry point — the same arm the
    /// mixed-device fix (forkwright/logismos#38) sits directly beside.
    /// A HIP device is nowhere in this fleet, so the GPU and `Mixed`
    /// arms cannot be driven end-to-end here; this pins the CPU-only
    /// arithmetic that CAN run, and CI confirms it on every push.
    #[test]
    fn both_cpu_rms_norm_computes_expected_row() {
        let x = f16_tensor(&[3.0, 4.0], &[1, 2]);
        let weight = f16_tensor(&[1.0, 1.0], &[2]);
        let out = rms_norm(&x, &weight, 0.0).expect("both-CPU rms_norm must still succeed");
        let got: Vec<f32> = out
            .to_host_f16()
            .expect("to_host_f16")
            .into_iter()
            .map(f16::to_f32)
            .collect();
        let rms = f32::midpoint(3.0_f32 * 3.0, 4.0 * 4.0);
        let rms = rms.sqrt();
        let want = [3.0 / rms, 4.0 / rms];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-2, "got {got:?}, want {want:?}");
        }
    }
}
