//! Flat-layout KV cache (Phase 2).
//!
//! One pre-allocated contiguous per-layer buffer per K and V, sized to
//! `max_seq_len * num_kv_heads * head_dim`. Append-on-write. Read
//! produces a fresh `taxis::Tensor` of shape
//! `[len, num_kv_heads * head_dim]`.
//!
//! The cache stores bytes rather than typed `Vec<T>` so one impl covers
//! every dtype. Conversion back to a typed tensor happens only at `get`
//! time through the dtype-dispatched branch.
//!
//! Paged + radix layouts (Phases 6 + 12) will replace the flat arrays
//! with block tables but keep the same `KvCache` trait contract.
//!
//! ## Byte-marshalling convention
//!
//! Every multi-byte dtype this crate stores (f32/f16/bf16/i32) is
//! marshalled **little-endian**, both directions: [`cpu_storage_bytes`]
//! writes it, the `chunks_to_*` functions read it back. This is stated
//! once, here — the write side used to reinterpret native-endian bytes
//! directly, which agreed with the little-endian readers only on a
//! little-endian host.

use std::borrow::Cow;

use taxis::{CpuStorage, DType, Shape, Tensor};

use crate::KvCache;
use crate::error::{Error, Result};

/// Shape + dtype invariants of a cache.
#[derive(Debug, Clone, Copy)]
pub struct CacheLayout {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads (after GQA reduction from Q heads).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Maximum context length this cache was sized for.
    pub max_seq_len: usize,
    /// Dtype of cached K and V tensors.
    pub dtype: DType,
}

impl CacheLayout {
    /// Row stride in bytes: bytes per (token, layer) row across all
    /// KV heads.
    #[must_use]
    pub(crate) fn row_bytes(&self) -> usize {
        let elems = self.num_kv_heads * self.head_dim;
        self.dtype.byte_count(elems)
    }

    /// Total byte count per K or V buffer, per layer.
    #[must_use]
    pub(crate) fn buffer_bytes(&self) -> usize {
        self.row_bytes() * self.max_seq_len
    }

    /// Row-width in element count (num_kv_heads × head_dim).
    #[must_use]
    pub fn row_elems(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// Validated view of a CPU-backed `[n_tokens, row_elems]` tensor's raw bytes.
struct TensorBytes<'t> {
    n_tokens: usize,
    bytes: Cow<'t, [u8]>,
}

/// Flat KV cache.
///
/// Invariants:
/// - `k_buffers.len() == v_buffers.len() == num_layers`.
/// - Each buffer is exactly `layout.buffer_bytes()` bytes long.
/// - `lens[layer]` is the number of rows written so far. Never
///   exceeds `layout.max_seq_len`.
pub struct FlatKvCache {
    layout: CacheLayout,
    k_buffers: Vec<Vec<u8>>,
    v_buffers: Vec<Vec<u8>>,
    lens: Vec<usize>,
}

impl FlatKvCache {
    /// Allocate a cache sized according to `layout`.
    #[must_use]
    pub fn new(layout: CacheLayout) -> Self {
        let buf_bytes = layout.buffer_bytes();
        let k_buffers = (0..layout.num_layers)
            .map(|_| vec![0u8; buf_bytes])
            .collect();
        let v_buffers = (0..layout.num_layers)
            .map(|_| vec![0u8; buf_bytes])
            .collect();
        Self {
            lens: vec![0; layout.num_layers],
            layout,
            k_buffers,
            v_buffers,
        }
    }

    /// Layout the cache was sized with.
    #[must_use]
    pub fn layout(&self) -> &CacheLayout {
        &self.layout
    }

    /// Validate and extract the per-token byte slice from a CPU-backed
    /// tensor with shape `[n_tokens, row_elems]`.
    fn tensor_as_bytes<'t>(&self, t: &'t Tensor) -> Result<TensorBytes<'t>> {
        if t.dtype() != self.layout.dtype {
            return Err(Error::DTypeMismatch {
                cache: self.layout.dtype,
                supplied: t.dtype(),
            });
        }
        let dims = t.dims();
        let (n_tokens, row_elems) = match dims {
            [n, e] => (*n, *e),
            other => {
                return Err(Error::ShapeMismatch {
                    msg: format!("expected rank-2 [n_tokens, kv_heads*head_dim], got {other:?}"),
                });
            }
        };
        if row_elems != self.layout.row_elems() {
            return Err(Error::ShapeMismatch {
                msg: format!(
                    "row_elems {} != cache row_elems {}",
                    row_elems,
                    self.layout.row_elems()
                ),
            });
        }
        let storage = t.cpu_storage().ok_or_else(|| Error::UnsupportedStorage {
            msg: "Phase-2 FlatKvCache only accepts CPU-backed tensors".into(),
        })?;
        let bytes = cpu_storage_bytes(storage)?;
        let expected = self.layout.dtype.byte_count(n_tokens * row_elems);
        if bytes.len() != expected {
            return Err(Error::ShapeMismatch {
                msg: format!(
                    "tensor byte length {} != expected {expected} (n_tokens={n_tokens}, \
                     row_elems={row_elems})",
                    bytes.len()
                ),
            });
        }
        Ok(TensorBytes { n_tokens, bytes })
    }

    fn check_layer(&self, layer_idx: usize) -> Result<()> {
        if layer_idx >= self.layout.num_layers {
            return Err(Error::LayerOutOfRange {
                layer_idx,
                num_layers: self.layout.num_layers,
            });
        }
        Ok(())
    }
}

impl KvCache for FlatKvCache {
    fn put(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        self.check_layer(layer_idx)?;
        let TensorBytes {
            n_tokens: n_k,
            bytes: k_bytes,
        } = self.tensor_as_bytes(k)?;
        let TensorBytes {
            n_tokens: n_v,
            bytes: v_bytes,
        } = self.tensor_as_bytes(v)?;
        if n_k != n_v {
            return Err(Error::ShapeMismatch {
                msg: format!("k n_tokens={n_k} != v n_tokens={n_v}"),
            });
        }
        let current = self.lens.get(layer_idx).copied().ok_or({
            Error::LayerOutOfRange {
                layer_idx,
                num_layers: self.layout.num_layers,
            }
        })?;
        if current + n_k > self.layout.max_seq_len {
            return Err(Error::LenOverflow {
                layer_idx,
                current,
                n_new: n_k,
                max_seq_len: self.layout.max_seq_len,
            });
        }
        let row_bytes = self.layout.row_bytes();
        let off = current * row_bytes;
        let end = off + n_k * row_bytes;
        let shape_err = || Error::ShapeMismatch {
            msg: format!(
                "layer {layer_idx} buffer overflow (off={off}, end={end}, \
                 buf_bytes={})",
                self.layout.buffer_bytes()
            ),
        };
        let num_layers = self.layout.num_layers;
        let k_buf = self
            .k_buffers
            .get_mut(layer_idx)
            .ok_or(Error::LayerOutOfRange {
                layer_idx,
                num_layers,
            })?;
        k_buf
            .get_mut(off..end)
            .ok_or_else(shape_err)?
            .copy_from_slice(&k_bytes);
        let v_buf = self
            .v_buffers
            .get_mut(layer_idx)
            .ok_or(Error::LayerOutOfRange {
                layer_idx,
                num_layers,
            })?;
        v_buf
            .get_mut(off..end)
            .ok_or_else(shape_err)?
            .copy_from_slice(&v_bytes);
        if let Some(slot) = self.lens.get_mut(layer_idx) {
            *slot = current + n_k;
        }
        Ok(())
    }

    fn get(&self, layer_idx: usize, len: usize) -> Result<(Tensor, Tensor)> {
        self.check_layer(layer_idx)?;
        let current = self.lens.get(layer_idx).copied().ok_or({
            Error::LayerOutOfRange {
                layer_idx,
                num_layers: self.layout.num_layers,
            }
        })?;
        if len > current {
            return Err(Error::ReadBeyondWritten {
                layer_idx,
                requested: len,
                current,
            });
        }
        let row_bytes = self.layout.row_bytes();
        let end = len * row_bytes;
        let layer_err = || Error::LayerOutOfRange {
            layer_idx,
            num_layers: self.layout.num_layers,
        };
        let read_err = || Error::ReadBeyondWritten {
            layer_idx,
            requested: len,
            current,
        };
        let k_slice = self
            .k_buffers
            .get(layer_idx)
            .ok_or_else(layer_err)?
            .get(..end)
            .ok_or_else(read_err)?;
        let v_slice = self
            .v_buffers
            .get(layer_idx)
            .ok_or_else(layer_err)?
            .get(..end)
            .ok_or_else(read_err)?;
        let shape = Shape::new(&[len, self.layout.row_elems()]);
        let k = cpu_tensor_from_bytes(self.layout.dtype, k_slice, shape.clone())?;
        let v = cpu_tensor_from_bytes(self.layout.dtype, v_slice, shape)?;
        Ok((k, v))
    }

    fn len_of(&self, layer_idx: usize) -> Option<usize> {
        // NEGATIVE-FIXTURE INJECTION (forkwright/logismos#49): reproduces
        // the pre-fix `unwrap_or(0)` behavior under the new signature, to
        // watch `len_of_distinguishes_out_of_range_from_unwritten` fail
        // in CI. Reverted before merge.
        Some(self.lens.get(layer_idx).copied().unwrap_or(0))
    }

    fn num_layers(&self) -> usize {
        self.layout.num_layers
    }

    fn reset(&mut self) {
        for l in &mut self.lens {
            *l = 0;
        }
    }
}

/// Little-endian byte view of `v` — the write-side half of this crate's
/// byte-marshalling convention (see the module doc). Zero-copy: on a
/// little-endian target, an element's native in-memory layout already IS
/// its little-endian encoding, so `_to_le` goes unused by construction —
/// kept in the signature so this and its big-endian sibling below share
/// one call-site shape.
#[cfg(target_endian = "little")]
fn le_bytes_of<T: Copy, const N: usize>(v: &[T], _to_le: impl Fn(T) -> [u8; N]) -> Cow<'_, [u8]> {
    // SAFETY: `T` is `Copy` + every bit pattern is valid
    // (f32/f16/bf16/i32/i8), and on this little-endian target the native
    // representation already equals the little-endian encoding `_to_le`
    // would produce, so this reinterpret is bit-for-bit identical to
    // calling `_to_le` on every element.
    let bytes =
        unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) };
    Cow::Borrowed(bytes)
}

/// Little-endian byte view of `v`, explicit-encode fallback for a
/// big-endian target — correctness here never depends on host
/// endianness (see the little-endian sibling above for the zero-copy
/// case, which covers every target this crate currently ships on).
#[cfg(not(target_endian = "little"))]
fn le_bytes_of<T: Copy, const N: usize>(v: &[T], to_le: impl Fn(T) -> [u8; N]) -> Cow<'_, [u8]> {
    let mut out = Vec::with_capacity(v.len() * N);
    for &x in v {
        out.extend_from_slice(&to_le(x));
    }
    Cow::Owned(out)
}

fn cpu_storage_bytes(s: &CpuStorage) -> Result<Cow<'_, [u8]>> {
    match s {
        CpuStorage::F32(v) => Ok(le_bytes_of(v, f32::to_le_bytes)),
        CpuStorage::F16(v) => Ok(le_bytes_of(v, half::f16::to_le_bytes)),
        CpuStorage::BF16(v) => Ok(le_bytes_of(v, half::bf16::to_le_bytes)),
        CpuStorage::I32(v) => Ok(le_bytes_of(v, i32::to_le_bytes)),
        CpuStorage::I8(v) => Ok(le_bytes_of(v, i8::to_le_bytes)),
        CpuStorage::U8(v) => Ok(Cow::Borrowed(v.as_slice())),
        _ => Err(Error::UnsupportedStorage {
            msg: "unsupported future CpuStorage variant".into(),
        }),
    }
}

fn cpu_tensor_from_bytes(dtype: DType, bytes: &[u8], shape: Shape) -> Result<Tensor> {
    let elem_count = shape.elem_count();
    let storage = match dtype {
        DType::F32 => CpuStorage::F32(chunks_to_f32(bytes, elem_count)?),
        DType::F16 => CpuStorage::F16(chunks_to_f16(bytes, elem_count)?),
        DType::BF16 => CpuStorage::BF16(chunks_to_bf16(bytes, elem_count)?),
        DType::I32 => CpuStorage::I32(chunks_to_i32(bytes, elem_count)?),
        DType::I8 => CpuStorage::I8(bytes_to_i8(bytes)),
        DType::U8 => CpuStorage::U8(bytes.to_vec()),
        other => {
            return Err(Error::Msg(format!(
                "dtype {other:?} not supported by Phase-2 FlatKvCache"
            )));
        }
    };
    Ok(Tensor::from_cpu(storage, shape))
}

/// Postcondition on every `chunks_exact`-based decoder: `chunks_exact`
/// silently drops a trailing partial chunk, so a byte length that isn't
/// a whole multiple of the dtype width would otherwise produce a `Vec`
/// shorter than `expected` — and `Tensor::from_cpu` performs no length
/// check of its own against the `Shape` it's handed, so that mismatch
/// would surface later as an out-of-bounds read, not a construction
/// error.
fn check_decoded_len(produced: usize, expected: usize, byte_len: usize) -> Result<()> {
    if produced != expected {
        return Err(Error::ShapeMismatch {
            msg: format!(
                "decoded {produced} elements from {byte_len} bytes, expected {expected} \
                 (dtype width does not evenly divide the supplied bytes)"
            ),
        });
    }
    Ok(())
}

fn bytes_to_i8(bytes: &[u8]) -> Vec<i8> {
    let mut out = Vec::with_capacity(bytes.len());
    for c in bytes.chunks_exact(1) {
        let mut arr = [0u8; 1];
        arr.copy_from_slice(c);
        // WHY(forkwright/logismos#42): a single byte has no endianness of
        // its own, so `from_ne_bytes` and `from_le_bytes` produce
        // identical output here today — but this reader sits beside
        // f32/i32/f16/bf16 readers that are all explicitly little-endian,
        // and a byte-order convention stated in three places and silently
        // exempted in a fourth is a defect waiting for a width change.
        // `from_le_bytes` makes the convention uniform and explicit.
        out.push(i8::from_le_bytes(arr));
    }
    out
}

fn chunks_to_f32(bytes: &[u8], elem: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(c);
        out.push(f32::from_le_bytes(b));
    }
    check_decoded_len(out.len(), elem, bytes.len())?;
    Ok(out)
}
fn chunks_to_i32(bytes: &[u8], elem: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(c);
        out.push(i32::from_le_bytes(b));
    }
    check_decoded_len(out.len(), elem, bytes.len())?;
    Ok(out)
}
fn chunks_to_f16(bytes: &[u8], elem: usize) -> Result<Vec<half::f16>> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(2) {
        let mut b = [0u8; 2];
        b.copy_from_slice(c);
        out.push(half::f16::from_le_bytes(b));
    }
    check_decoded_len(out.len(), elem, bytes.len())?;
    Ok(out)
}
fn chunks_to_bf16(bytes: &[u8], elem: usize) -> Result<Vec<half::bf16>> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(2) {
        let mut b = [0u8; 2];
        b.copy_from_slice(c);
        out.push(half::bf16::from_le_bytes(b));
    }
    check_decoded_len(out.len(), elem, bytes.len())?;
    Ok(out)
}

#[cfg(test)]
#[path = "flat_tests.rs"]
mod tests;
