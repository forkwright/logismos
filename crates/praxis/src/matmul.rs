//! `praxis::matmul` — composed matmul over `taxis::Tensor`.

use std::ffi::c_void;

use taxis::{DType, Shape, Tensor};

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
    // to the CPU reference path.
    match (a.hip_storage(), b.hip_storage()) {
        (Some(a_hip), Some(b_hip)) => {
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
                // buffer.
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
        _ => {
            // CPU fallback — both on CPU.
            let a_host = a.to_host_f16()?;
            let b_host = b.to_host_f16()?;
            let out = kernels::matmul::cpu::matmul_fp16_ref(&a_host, &b_host, m, n, ka);
            Ok(Tensor::from_cpu(
                taxis::CpuStorage::F16(out),
                Shape::new(&[m, n]),
            ))
        }
    }
}
