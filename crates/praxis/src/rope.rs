//! `praxis::rope_apply` — rotary position embedding.

use std::ffi::{c_int, c_void};
use std::sync::{Arc, Mutex, PoisonError};

use hipcore::{Device, DeviceBuffer};
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
    /// Device-resident upload, keyed by device ordinal. `rope_apply`
    /// re-uploads only on the first call for a given device (or after a
    /// different device is passed), instead of on every call.
    device_cache: Mutex<Option<(c_int, Arc<DeviceBuffer<u8>>)>>,
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
            device_cache: Mutex::new(None),
        }
    }

    /// Device-resident cos/sin table for `device`, uploading once and
    /// reusing the allocation on subsequent calls for the same device.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] if the upload fails.
    fn device_buffer(&self, device: &Device) -> Result<Arc<DeviceBuffer<u8>>> {
        // WARNING: do not panic-unwrap a mutex poison here — `unwrap_used`
        // is a deny-level lint in this workspace. Recovering the guard is
        // correct: a panic while holding this lock leaves the cached
        // upload's *content* untouched (only the cache slot's
        // book-keeping could be mid write), so treating a poisoned lock
        // as a normal guard is safe.
        let mut cache = self
            .device_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((ordinal, buf)) = cache.as_ref()
            && *ordinal == device.ordinal()
        {
            return Ok(Arc::clone(buf));
        }
        let cs_bytes: &[u8] = unsafe {
            // SAFETY: `f32` is BytePod; slice covers `data.len()` f32s.
            std::slice::from_raw_parts(
                self.data.as_ptr().cast::<u8>(),
                self.data.len() * size_of::<f32>(),
            )
        };
        let buf = Arc::new(DeviceBuffer::<u8>::from_host(device, cs_bytes)?);
        *cache = Some((device.ordinal(), Arc::clone(&buf)));
        Ok(buf)
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
    // WHY: the pair-rotation kernel and its CPU reference both index
    // `2 * pair_idx` / `2 * pair_idx + 1` off `head_dim / 2`. An odd
    // `head_dim` truncates that division, silently dropping the final
    // element instead of rotating it — reject it here rather than
    // letting a malformed checkpoint produce quietly wrong encoding.
    if !head_dim.is_multiple_of(2) {
        return Err(Error::Invalid {
            op: "rope_apply",
            msg: format!("head_dim must be even, got {head_dim}"),
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

            let cs_dev = table.device_buffer(device)?;

            crate::stream_pool::POOL.with_stream(device, |stream| {
                // SAFETY: device pointers valid; sizes verified above.
                unsafe {
                    kernels::rope::launch_rope_fp16_in_place(
                        out_hip.as_mut_device_ptr().cast::<c_void>(),
                        cs_dev.as_device_ptr().cast::<c_void>(),
                        dim_i32(batch, "batch")?,
                        dim_i32(seq, "seq")?,
                        dim_i32(heads, "heads")?,
                        dim_i32(head_dim, "head_dim")?,
                        stream,
                    )?;
                }
                stream.synchronize()?;
                Ok(())
            })?;
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

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions use expect() directly")]

    use half::f16;
    use taxis::{CpuStorage, Shape};

    use super::*;

    #[test]
    fn rope_apply_rejects_odd_head_dim() {
        let (batch, seq, heads, head_dim) = (1_usize, 2_usize, 1_usize, 5_usize);
        let qk = Tensor::from_cpu(
            CpuStorage::F16(vec![f16::from_f32(0.0); batch * seq * heads * head_dim]),
            Shape::new(&[batch, seq, heads, head_dim]),
        );
        // Bypass `CosSinTable::new` — its CPU table builder
        // debug_asserts an even `head_dim` itself, which would panic
        // before `rope_apply`'s own validation ever ran. Constructing
        // the table directly isolates the check under test.
        let table = CosSinTable {
            data: vec![0.0; seq * head_dim],
            seq,
            head_dim,
            device_cache: Mutex::new(None),
        };

        let err = rope_apply(&qk, &table).expect_err("odd head_dim must be rejected");

        assert!(
            matches!(&err, Error::Invalid { op: "rope_apply", msg } if msg.contains("even")),
            "expected an even-head_dim Invalid error, got {err:?}"
        );
    }
}
