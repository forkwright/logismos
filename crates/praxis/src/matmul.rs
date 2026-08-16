//! `praxis::matmul` — composed matmul over `taxis::Tensor`.

use std::ffi::c_void;

use taxis::{DType, Shape, Tensor};

use crate::device_dispatch::{DevicePlacement, classify_placement};
use crate::error::{Error, Result};

fn dim_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Invalid {
        op: "matmul",
        msg: format!("{name} dimension exceeds i32: {value}"),
    })
}

/// `D = A @ B` where A is `(M, K)` and B is `(K, N)`. fp16 in, fp16 out.
///
/// Phase 1 shape contract: both inputs are 2-D, contiguous, on the same
/// HIP device, and dtype `F16`. Broadcasting-over-batch-dims and
/// mixed-precision paths land in Phase 2 / 3.
///
/// # Errors
///
/// [`Error::Invalid`] on shape / dtype mismatch; [`Error::Kernel`] on
/// launch failure.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() != DType::F16 || b.dtype() != DType::F16 {
        return Err(Error::Invalid {
            op: "matmul",
            msg: format!(
                "Phase 1 supports F16 inputs; got A={:?}, B={:?}",
                a.dtype(),
                b.dtype()
            ),
        });
    }
    if a.dims().len() != 2 || b.dims().len() != 2 {
        return Err(Error::Invalid {
            op: "matmul",
            msg: format!(
                "Phase 1 requires 2-D inputs; got A shape {:?}, B shape {:?}",
                a.dims(),
                b.dims()
            ),
        });
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Invalid {
            op: "matmul",
            msg: "inputs must be contiguous in Phase 1".into(),
        });
    }
    let a_dims = a.dims();
    let b_dims = b.dims();
    let (Some(&m), Some(&ka)) = (a_dims.first(), a_dims.get(1)) else {
        return Err(Error::Invalid {
            op: "matmul",
            msg: "A rank changed during validation".into(),
        });
    };
    let (Some(&kb), Some(&n)) = (b_dims.first(), b_dims.get(1)) else {
        return Err(Error::Invalid {
            op: "matmul",
            msg: "B rank changed during validation".into(),
        });
    };
    if ka != kb {
        return Err(Error::Invalid {
            op: "matmul",
            msg: format!("inner dims mismatch: A={ka}, B={kb}"),
        });
    }

    // Phase 1 dispatch: if inputs are on HIP, use the WMMA kernel for
    // 16-aligned shapes else the naive kernel. CPU tensors fall back
    // to the CPU reference path. A mixed HIP/CPU pair is rejected —
    // see `DevicePlacement::Mixed` below.
    let placement = classify_placement(a.hip_storage().is_some(), b.hip_storage().is_some());
    match placement {
        DevicePlacement::BothHip => {
            let (Some(a_hip), Some(b_hip)) = (a.hip_storage(), b.hip_storage()) else {
                return Err(Error::Invalid {
                    op: "matmul",
                    msg: "device-pair classification invariant violated: BothHip without two HIP operands".into(),
                });
            };
            let device = a_hip.device();
            if device.ordinal() != b_hip.device().ordinal() {
                return Err(Error::Invalid {
                    op: "matmul",
                    msg: "A and B must live on the same device".into(),
                });
            }
            let out = Tensor::zeros_hip(device, DType::F16, Shape::new(&[m, n]))?;
            let out_hip = out.hip_storage().ok_or_else(|| Error::Invalid {
                op: "matmul",
                msg: "zeros_hip did not return a HIP tensor".into(),
            })?;
            let variant = if m % 16 == 0 && n % 16 == 0 && ka % 16 == 0 {
                kernels::matmul::Variant::Wmma
            } else {
                kernels::matmul::Variant::Naive
            };

            crate::stream_pool::POOL.with_stream(device, |stream| {
                // SAFETY: pointers originate from our own device
                // allocations and are valid on `stream`'s device for the
                // duration of the launch; shapes derived from the tensor
                // metadata; no aliasing between A, B, and the fresh out
                // buffer. `out` was just allocated above and not yet
                // shared with any other `Tensor` handle, so `out_hip`'s
                // `as_mut_device_ptr` obligation (no other live pointer to
                // this allocation) holds by construction.
                unsafe {
                    kernels::matmul::launch_matmul_fp16(
                        variant,
                        a_hip.as_device_ptr().cast::<c_void>(),
                        b_hip.as_device_ptr().cast::<c_void>(),
                        out_hip.as_mut_device_ptr().cast::<c_void>(),
                        dim_i32(m, "M")?,
                        dim_i32(n, "N")?,
                        dim_i32(ka, "K")?,
                        stream,
                    )?;
                }
                stream.synchronize()?;
                Ok(())
            })?;
            Ok(out)
        }
        DevicePlacement::BothCpu => {
            let a_host = a.to_host_f16()?;
            let b_host = b.to_host_f16()?;
            let out = kernels::matmul::cpu::matmul_fp16_ref(&a_host, &b_host, m, n, ka);
            Ok(Tensor::from_cpu(
                taxis::CpuStorage::F16(out),
                Shape::new(&[m, n]),
            ))
        }
        // WHY(forkwright/logismos#39): a mixed pair used to fall through
        // the old wildcard arm — commented "CPU fallback — both on
        // CPU" while structurally matching every non-both-HIP
        // combination — silently transferring the HIP operand to host
        // with no error and no diagnostic. A and B must both be on CPU
        // or both on the same HIP device; mismatched placement is a
        // caller bug and must fail loudly (CLAUDE.md:77, AGENTS.md:29
        // — no silent CPU fallbacks).
        DevicePlacement::Mixed => Err(Error::Invalid {
            op: "matmul",
            msg: "A and B must both be on CPU or both on the same HIP device".into(),
        }),
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

    use super::matmul;

    fn f16_tensor(values: &[f32], shape: &[usize]) -> Tensor {
        let data: Vec<f16> = values.iter().copied().map(f16::from_f32).collect();
        Tensor::from_cpu(CpuStorage::F16(data), Shape::new(shape))
    }

    /// WHY: exercises the `DevicePlacement::BothCpu` arm through the
    /// real public `matmul` entry point — the same arm the mixed-device
    /// fix (forkwright/logismos#39) sits directly beside. A HIP device
    /// is nowhere in this fleet, so the GPU and `Mixed` arms cannot be
    /// driven end-to-end here; this pins the CPU-only arithmetic that
    /// CAN run, and CI confirms it on every push.
    #[test]
    fn both_cpu_matmul_computes_expected_product() {
        let a = f16_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let identity = f16_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let out = matmul(&a, &identity).expect("both-CPU matmul must still succeed");
        let got = out.to_host_f16().expect("to_host_f16");
        let want: Vec<f16> = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(got, want, "A @ I must equal A");
    }
}
