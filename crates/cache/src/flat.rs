//! Flat-layout KV cache (Phase 2).
//!
//! One pre-allocated contiguous per-layer buffer per K and V, sized to
//! `max_seq_len * num_kv_heads * head_dim`. Append-on-write. Read
//! produces a fresh `taxis::Tensor` of shape
//! `[len, num_kv_heads * head_dim]`.
//!
//! The cache stores bytes rather than typed `Vec<T>` and marshals the
//! supported f32/f16/bf16/i32/i8/u8 storage variants. F8 and I4 tensor
//! marshalling is unsupported and returns a typed refusal.
//!
//! Native hybrid execution uses separate execution-private paged KV. This
//! legacy cache retains its flat tensor-based [`KvCache`] contract.
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
use crate::error::{
    DTypeMismatchSnafu, FlatAllocationSnafu, FlatArithmeticSnafu, FlatZeroDimensionSnafu,
    LayerOutOfRangeSnafu, LenOverflowSnafu, MsgSnafu, ReadBeyondWrittenSnafu, Result,
    ShapeMismatchSnafu, UnsupportedStorageSnafu,
};

/// Shape + dtype invariants of a cache.
#[derive(Debug, Clone, Copy)]
pub struct CacheLayout {
    /// Number of transformer layers.
    num_layers: usize,
    /// Maximum context length this cache was sized for.
    max_seq_len: usize,
    /// Dtype of cached K and V tensors.
    dtype: DType,
    row_elems: usize,
    row_bytes: usize,
    buffer_bytes: usize,
}

impl CacheLayout {
    /// Construct checked immutable cache geometry.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::FlatZeroDimension`] for a zero configured
    /// dimension, [`crate::Error::FlatArithmetic`] when any row, per-layer,
    /// or all-layer K/V backing extent overflows, and [`crate::Error::Taxis`]
    /// when the dtype byte extent cannot be represented.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> Result<Self> {
        for (field, value) in [
            ("num_layers", num_layers),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("max_seq_len", max_seq_len),
        ] {
            if value == 0 {
                return FlatZeroDimensionSnafu { field }.fail();
            }
        }
        let row_elems = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
            FlatArithmeticSnafu {
                operation: "row elements",
            }
            .build()
        })?;
        let row_bytes = dtype.checked_byte_count(row_elems)?;
        let buffer_bytes = row_bytes.checked_mul(max_seq_len).ok_or_else(|| {
            FlatArithmeticSnafu {
                operation: "per-layer buffer bytes",
            }
            .build()
        })?;
        buffer_bytes
            .checked_mul(num_layers)
            .and_then(|per_kind| per_kind.checked_mul(2))
            .ok_or_else(|| {
                FlatArithmeticSnafu {
                    operation: "all-layer K/V backing bytes",
                }
                .build()
            })?;
        Ok(Self {
            num_layers,
            max_seq_len,
            dtype,
            row_elems,
            row_bytes,
            buffer_bytes,
        })
    }

    /// Row stride in bytes: bytes per (token, layer) row across all
    /// KV heads.
    #[must_use]
    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Total byte count per K or V buffer, per layer.
    #[must_use]
    pub(crate) fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Row-width in element count (num_kv_heads × head_dim).
    #[must_use]
    pub fn row_elems(&self) -> usize {
        self.row_elems
    }

    /// Number of configured transformer layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// Validated view of a CPU-backed `[n_tokens, row_elems]` tensor's raw bytes.
struct TensorBytes<'t> {
    n_tokens: usize,
    bytes: Cow<'t, [u8]>,
}

struct AppendSpan {
    next_len: usize,
    offset: usize,
    end: usize,
}

fn append_span(
    layer_idx: usize,
    current: usize,
    append: usize,
    row_bytes: usize,
    max_seq_len: usize,
) -> Result<AppendSpan> {
    let next_len = current.checked_add(append).ok_or_else(|| {
        FlatArithmeticSnafu {
            operation: "append length",
        }
        .build()
    })?;
    if next_len > max_seq_len {
        return LenOverflowSnafu {
            layer_idx,
            current,
            n_new: append,
            max_seq_len,
        }
        .fail();
    }
    let offset = current.checked_mul(row_bytes).ok_or_else(|| {
        FlatArithmeticSnafu {
            operation: "append offset",
        }
        .build()
    })?;
    let append_bytes = append.checked_mul(row_bytes).ok_or_else(|| {
        FlatArithmeticSnafu {
            operation: "append byte count",
        }
        .build()
    })?;
    let end = offset.checked_add(append_bytes).ok_or_else(|| {
        FlatArithmeticSnafu {
            operation: "append end offset",
        }
        .build()
    })?;
    Ok(AppendSpan {
        next_len,
        offset,
        end,
    })
}

fn allocate_buffers(count: usize, bytes: usize, target: &'static str) -> Result<Vec<Vec<u8>>> {
    let mut buffers = Vec::new();
    buffers
        .try_reserve_exact(count)
        .map_err(|source| FlatAllocationSnafu { target, source }.build())?;
    for _ in 0..count {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(bytes)
            .map_err(|source| FlatAllocationSnafu { target, source }.build())?;
        buffer.resize(bytes, 0);
        buffers.push(buffer);
    }
    Ok(buffers)
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
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::FlatAllocation`] when any K, V, or length
    /// backing reservation fails. `layout` has already checked the aggregate
    /// K/V extent before this method begins allocating.
    pub fn new(layout: CacheLayout) -> Result<Self> {
        let buffer_bytes = layout.buffer_bytes();
        let k_buffers = allocate_buffers(layout.num_layers, buffer_bytes, "K")?;
        let v_buffers = allocate_buffers(layout.num_layers, buffer_bytes, "V")?;
        let mut lens = Vec::new();
        lens.try_reserve_exact(layout.num_layers)
            .map_err(|source| {
                FlatAllocationSnafu {
                    target: "length",
                    source,
                }
                .build()
            })?;
        lens.resize(layout.num_layers, 0);
        Ok(Self {
            lens,
            layout,
            k_buffers,
            v_buffers,
        })
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
            return DTypeMismatchSnafu {
                cache: self.layout.dtype,
                supplied: t.dtype(),
            }
            .fail();
        }
        let dims = t.dims();
        let (n_tokens, row_elems) = match dims {
            [n, e] => (*n, *e),
            other => {
                return ShapeMismatchSnafu {
                    msg: format!("expected rank-2 [n_tokens, kv_heads*head_dim], got {other:?}"),
                }
                .fail();
            }
        };
        if row_elems != self.layout.row_elems() {
            return ShapeMismatchSnafu {
                msg: format!(
                    "row_elems {} != cache row_elems {}",
                    row_elems,
                    self.layout.row_elems()
                ),
            }
            .fail();
        }
        let storage = t.cpu_storage().ok_or_else(|| {
            UnsupportedStorageSnafu {
                msg: "Phase-2 FlatKvCache only accepts CPU-backed tensors",
            }
            .build()
        })?;
        let bytes = cpu_storage_bytes(storage)?;
        let expected_elements = n_tokens.checked_mul(row_elems).ok_or_else(|| {
            FlatArithmeticSnafu {
                operation: "tensor elements",
            }
            .build()
        })?;
        let expected = self.layout.dtype.checked_byte_count(expected_elements)?;
        if bytes.len() != expected {
            return ShapeMismatchSnafu {
                msg: format!(
                    "tensor byte length {} != expected {expected} (n_tokens={n_tokens}, \
                     row_elems={row_elems})",
                    bytes.len()
                ),
            }
            .fail();
        }
        Ok(TensorBytes { n_tokens, bytes })
    }

    fn check_layer(&self, layer_idx: usize) -> Result<()> {
        if layer_idx >= self.layout.num_layers {
            return LayerOutOfRangeSnafu {
                layer_idx,
                num_layers: self.layout.num_layers,
            }
            .fail();
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
            return ShapeMismatchSnafu {
                msg: format!("k n_tokens={n_k} != v n_tokens={n_v}"),
            }
            .fail();
        }
        let current = self.lens.get(layer_idx).copied().ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers: self.layout.num_layers,
            }
            .build()
        })?;
        let row_bytes = self.layout.row_bytes();
        let span = append_span(layer_idx, current, n_k, row_bytes, self.layout.max_seq_len)?;
        let AppendSpan {
            next_len: next,
            offset: off,
            end,
        } = span;
        let buffer_bytes = self.layout.buffer_bytes();
        let shape_err = || {
            ShapeMismatchSnafu {
                msg: format!(
                    "layer {layer_idx} buffer overflow (off={off}, end={end}, \
                     buf_bytes={buffer_bytes})"
                ),
            }
            .build()
        };
        let num_layers = self.layout.num_layers;
        let (k_buffers, v_buffers) = (&mut self.k_buffers, &mut self.v_buffers);
        let k_buf = k_buffers.get_mut(layer_idx).ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers,
            }
            .build()
        })?;
        let v_buf = v_buffers.get_mut(layer_idx).ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers,
            }
            .build()
        })?;
        let k_destination = k_buf.get_mut(off..end).ok_or_else(shape_err)?;
        let v_destination = v_buf.get_mut(off..end).ok_or_else(shape_err)?;
        k_destination.copy_from_slice(&k_bytes);
        v_destination.copy_from_slice(&v_bytes);
        if let Some(slot) = self.lens.get_mut(layer_idx) {
            *slot = next;
        }
        Ok(())
    }

    fn get(&self, layer_idx: usize, len: usize) -> Result<(Tensor, Tensor)> {
        self.check_layer(layer_idx)?;
        let current = self.lens.get(layer_idx).copied().ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers: self.layout.num_layers,
            }
            .build()
        })?;
        if len > current {
            return ReadBeyondWrittenSnafu {
                layer_idx,
                requested: len,
                current,
            }
            .fail();
        }
        let row_bytes = self.layout.row_bytes();
        let end = len.checked_mul(row_bytes).ok_or_else(|| {
            FlatArithmeticSnafu {
                operation: "read end offset",
            }
            .build()
        })?;
        let layer_err = || {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers: self.layout.num_layers,
            }
            .build()
        };
        let read_err = || {
            ReadBeyondWrittenSnafu {
                layer_idx,
                requested: len,
                current,
            }
            .build()
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
        self.lens.get(layer_idx).copied()
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
        _ => UnsupportedStorageSnafu {
            msg: "unsupported future CpuStorage variant",
        }
        .fail(),
    }
}

fn cpu_tensor_from_bytes(dtype: DType, bytes: &[u8], shape: Shape) -> Result<Tensor> {
    let elem_count = shape.checked_elem_count().map_err(|_| {
        FlatArithmeticSnafu {
            operation: "decoded tensor element count",
        }
        .build()
    })?;
    let storage = match dtype {
        DType::F32 => CpuStorage::F32(chunks_to_f32(bytes, elem_count)?),
        DType::F16 => CpuStorage::F16(chunks_to_f16(bytes, elem_count)?),
        DType::BF16 => CpuStorage::BF16(chunks_to_bf16(bytes, elem_count)?),
        DType::I32 => CpuStorage::I32(chunks_to_i32(bytes, elem_count)?),
        DType::I8 => CpuStorage::I8(bytes_to_i8(bytes)),
        DType::U8 => CpuStorage::U8(bytes.to_vec()),
        other => {
            return MsgSnafu {
                message: format!("dtype {other:?} not supported by Phase-2 FlatKvCache"),
            }
            .fail();
        }
    };
    Tensor::from_cpu(storage, shape).map_err(crate::Error::from)
}

/// Reject a decoded-length mismatch at the serialized-byte boundary.
///
/// `chunks_exact` omits partial trailing chunks. Refusing here preserves
/// byte-decoding context before the tensor constructor checks shape/storage
/// compatibility independently.
fn check_decoded_len(produced: usize, expected: usize, byte_len: usize) -> Result<()> {
    if produced != expected {
        return ShapeMismatchSnafu {
            msg: format!(
                "decoded {produced} elements from {byte_len} bytes, expected {expected} \
                 (dtype width does not evenly divide the supplied bytes)"
            ),
        }
        .fail();
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
