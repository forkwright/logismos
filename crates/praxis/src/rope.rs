//! `praxis::rope_apply` — rotary position embedding.

use std::ffi::c_void;

use hipcore::{Device, DeviceBuffer, Stream};
use taxis::{DType, Tensor};

use crate::error::{Error, Result};

fn dim_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Invalid {
        op: "rope_apply",
        msg: format!("{name} dimension exceeds i32: {value}"),
    })
}

/// Precomputed rotary table in fp32. Layout: `(seq, head_dim)` row-major
/// with interleaved (cos, sin) per pair along `head_dim`.
pub struct CosSinTable {
    /// Flat data.
    pub data: Vec<f32>,
    /// Max sequence length covered.
    pub seq: usize,
    /// Head dimension; must be even.
    pub head_dim: usize,
}

impl CosSinTable {
    /// Build a fresh rotary table using the standard Llama-style
    /// formula with base `theta`.
    #[must_use]
    pub fn new(seq: usize, head_dim: usize, theta: f32) -> Self {
        Self {
            data: kernels::rope::cpu::build_cos_sin_table(seq, head_dim, theta),
            seq,
            head_dim,
        }
    }
}

/// In-place RoPE on `qk` of shape `(batch, seq, heads, head_dim)` fp16.
///
/// Returns a new `Tensor` with the rotated values (the input is not
/// mutated at the `Tensor` level — Phase 1 `Tensor` is always cloned
/// to a fresh allocation).
///
/// # Errors
///
/// See [`Error::Invalid`] and propagated kernel errors.
pub fn rope_apply(qk: &Tensor, table: &CosSinTable) -> Result<Tensor> {
    if qk.dtype() != DType::F16 {
        return Err(Error::Invalid {
            op: "rope_apply",
            msg: "F16 only in Phase 1".into(),
        });
    }
    if qk.dims().len() != 4 {
        return Err(Error::Invalid {
            op: "rope_apply",
            msg: format!("expected (B, S, H, D); got {:?}", qk.dims()),
        });
    }
    let qk_dims = qk.dims();
    let (Some(&batch), Some(&seq), Some(&heads), Some(&head_dim)) = (
        qk_dims.first(),
        qk_dims.get(1),
        qk_dims.get(2),
        qk_dims.get(3),
    ) else {
        return Err(Error::Invalid {
            op: "rope_apply",
            msg: "input rank changed during validation".into(),
        });
    };
    if head_dim != table.head_dim || seq > table.seq {
        return Err(Error::Invalid {
            op: "rope_apply",
            msg: format!(
                "table shape ({}x{}) incompatible with tensor seq={seq}, head_dim={head_dim}",
                table.seq, table.head_dim
            ),
        });
    }

    match qk.hip_storage() {
        Some(qk_hip) => {
            let device = qk_hip.device();
            // Clone qk into a fresh device allocation (rope is
            // "in-place" at the kernel level, but the `Tensor` layer
            // preserves immutability).
            let out = clone_hip_tensor(qk, device)?;
            let out_hip = out.hip_storage().ok_or_else(|| Error::Invalid {
                op: "rope_apply",
                msg: "clone did not return HIP".into(),
            })?;

            // Upload the cos_sin table.
            let cs_bytes: &[u8] = unsafe {
                // SAFETY: `f32` is BytePod; slice is valid.
                std::slice::from_raw_parts(table.data.as_ptr().cast::<u8>(), table.data.len() * 4)
            };
            let cs_dev = DeviceBuffer::<u8>::from_host(device, cs_bytes)?;

            let stream = Stream::new(device)?;
            // SAFETY: device pointers valid; sizes verified above.
            unsafe {
                kernels::rope::launch_rope_fp16_in_place(
                    out_hip.as_mut_device_ptr().cast::<c_void>(),
                    cs_dev.as_device_ptr().cast::<c_void>(),
                    dim_i32(batch, "batch")?,
                    dim_i32(seq, "seq")?,
                    dim_i32(heads, "heads")?,
                    dim_i32(head_dim, "head_dim")?,
                    &stream,
                )?;
            }
            stream.synchronize()?;
            Ok(out)
        }
        None => {
            let mut host = qk.to_host_f16()?;
            kernels::rope::cpu::rope_apply_fp16_ref(
                &mut host,
                &table.data,
                batch,
                seq,
                heads,
                head_dim,
            );
            Ok(Tensor::from_cpu(
                taxis::CpuStorage::F16(host),
                qk.shape().clone(),
            ))
        }
    }
}

/// Deep copy a HIP-backed tensor into a freshly allocated tensor on
/// the same device. Uses a host round-trip; good enough for Phase 1.
fn clone_hip_tensor(t: &Tensor, device: &Device) -> Result<Tensor> {
    match t.dtype() {
        DType::F16 => {
            let host = t.to_host_f16()?;
            Ok(Tensor::from_host_f16(device, &host, t.shape().clone())?)
        }
        DType::F32 => {
            let host = t.to_host_f32()?;
            Ok(Tensor::from_host_f32(device, &host, t.shape().clone())?)
        }
        other => Err(Error::Invalid {
            op: "clone_hip_tensor",
            msg: format!("unsupported dtype {other:?}"),
        }),
    }
}
