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

use std::{borrow::Cow, ops::Range};

use snafu::ResultExt;
use taxis::{CpuStorage, DType, Shape, Tensor};

use crate::KvCache;
use crate::error::{
    AllocationSnafu, DTypeMismatchSnafu, GeometryOverflowSnafu, LayerOutOfRangeSnafu,
    LenOverflowSnafu, MsgSnafu, ReadBeyondWrittenSnafu, Result, ShapeMismatchSnafu,
    UnsupportedStorageSnafu,
};

/// Shape + dtype invariants of a cache.
#[derive(Debug, Clone)]
pub struct CacheLayout {
    num_layers: usize,
    max_seq_len: usize,
    dtype: DType,
    row_elems: usize,
    row_bytes: usize,
    buffer_bytes: usize,
}

impl CacheLayout {
    /// Validate cache geometry and retain each allocation-derived quantity.
    ///
    /// Zero layers, heads, widths, and context are valid empty domains. They
    /// remain distinct from nonzero geometry that cannot be represented.
    ///
    /// # Errors
    ///
    /// [`crate::Error::GeometryOverflow`] or [`crate::Error::Taxis`] when
    /// the requested geometry cannot be represented in `usize` bytes.
    pub fn try_new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
    ) -> Result<Self> {
        let row_elems = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "CacheLayout::try_new",
                msg: format!("num_kv_heads {num_kv_heads} × head_dim {head_dim} overflows"),
            }
            .build()
        })?;
        let row_bytes = dtype.byte_count(row_elems)?;
        let buffer_bytes = row_bytes.checked_mul(max_seq_len).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "CacheLayout::try_new",
                msg: format!("row_bytes {row_bytes} × max_seq_len {max_seq_len} overflows"),
            }
            .build()
        })?;
        let per_kind_bytes = buffer_bytes.checked_mul(num_layers).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "CacheLayout::try_new",
                msg: format!("buffer_bytes {buffer_bytes} × num_layers {num_layers} overflows"),
            }
            .build()
        })?;
        per_kind_bytes.checked_mul(2).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "CacheLayout::try_new",
                msg: "combined K/V allocation geometry overflows".to_string(),
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

    /// Number of transformer layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Maximum context length this cache was sized for.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Dtype of cached K and V tensors.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
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
}

/// Validated view of a CPU-backed `[n_tokens, row_elems]` tensor's raw bytes.
struct TensorBytes<'t> {
    n_tokens: usize,
    bytes: Cow<'t, [u8]>,
}

/// Checked destination geometry for one atomic cache append.
struct AppendGeometry {
    next_len: usize,
    byte_range: Range<usize>,
}

/// Flat KV cache.
///
/// Invariants:
/// - `k_buffers.len() == v_buffers.len() == num_layers`.
/// - Each buffer is exactly `layout.buffer_bytes()` bytes long.
/// - `lens[layer]` is the number of rows written so far. Never
///   exceeds `layout.max_seq_len()`.
pub struct FlatKvCache {
    layout: CacheLayout,
    k_buffers: Vec<Vec<u8>>,
    v_buffers: Vec<Vec<u8>>,
    lens: Vec<usize>,
}

impl FlatKvCache {
    /// Allocate a cache sized according to validated `layout`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Allocation`] when a validated cache cannot reserve
    /// memory. Callers migrating from the former infallible API must handle
    /// this result; no allocation path silently wraps or substitutes empty
    /// buffers.
    pub fn try_new(layout: CacheLayout) -> Result<Self> {
        let mut k_buffers: Vec<Vec<u8>> = try_reserve_vec(layout.num_layers(), "K buffer list")?;
        let mut v_buffers: Vec<Vec<u8>> = try_reserve_vec(layout.num_layers(), "V buffer list")?;
        let mut lens: Vec<usize> = try_reserve_vec(layout.num_layers(), "length list")?;
        for _ in 0..layout.num_layers() {
            k_buffers.push(allocate_zeroed(layout.buffer_bytes(), "K buffer")?);
            v_buffers.push(allocate_zeroed(layout.buffer_bytes(), "V buffer")?);
            lens.push(0);
        }
        Ok(Self {
            layout,
            k_buffers,
            v_buffers,
            lens,
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
        if t.dtype() != self.layout.dtype() {
            return DTypeMismatchSnafu {
                cache: self.layout.dtype(),
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
        let elems = n_tokens.checked_mul(row_elems).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "FlatKvCache::tensor_as_bytes",
                msg: format!("n_tokens {n_tokens} × row_elems {row_elems} overflows"),
            }
            .build()
        })?;
        let expected = self.layout.dtype().byte_count(elems)?;
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
        if layer_idx >= self.layout.num_layers() {
            return LayerOutOfRangeSnafu {
                layer_idx,
                num_layers: self.layout.num_layers(),
            }
            .fail();
        }
        Ok(())
    }

    fn append_geometry(
        &self,
        layer_idx: usize,
        current: usize,
        n_tokens: usize,
    ) -> Result<AppendGeometry> {
        let next_len = current.checked_add(n_tokens).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "FlatKvCache::put",
                msg: format!("current {current} + n_tokens {n_tokens} overflows"),
            }
            .build()
        })?;
        if next_len > self.layout.max_seq_len() {
            return LenOverflowSnafu {
                layer_idx,
                current,
                n_new: n_tokens,
                max_seq_len: self.layout.max_seq_len(),
            }
            .fail();
        }
        let row_bytes = self.layout.row_bytes();
        let off = current.checked_mul(row_bytes).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "FlatKvCache::put",
                msg: format!("current {current} × row_bytes {row_bytes} overflows"),
            }
            .build()
        })?;
        let write_bytes = n_tokens.checked_mul(row_bytes).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "FlatKvCache::put",
                msg: format!("n_tokens {n_tokens} × row_bytes {row_bytes} overflows"),
            }
            .build()
        })?;
        let end = off.checked_add(write_bytes).ok_or_else(|| {
            GeometryOverflowSnafu {
                op: "FlatKvCache::put",
                msg: "write range end overflows".to_string(),
            }
            .build()
        })?;
        Ok(AppendGeometry {
            next_len,
            byte_range: off..end,
        })
    }

    fn write_append(
        &mut self,
        layer_idx: usize,
        geometry: AppendGeometry,
        k_bytes: &[u8],
        v_bytes: &[u8],
    ) -> Result<()> {
        let AppendGeometry {
            next_len,
            byte_range,
        } = geometry;
        let range_start = byte_range.start;
        let range_end = byte_range.end;
        let buffer_err = || {
            ShapeMismatchSnafu {
                msg: format!(
                    "layer {layer_idx} buffer overflow (off={range_start}, end={range_end}, \
                     buf_bytes={})",
                    self.layout.buffer_bytes()
                ),
            }
            .build()
        };
        let num_layers = self.layout.num_layers();
        let k_buf = self.k_buffers.get_mut(layer_idx).ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers,
            }
            .build()
        })?;
        let v_buf = self.v_buffers.get_mut(layer_idx).ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers,
            }
            .build()
        })?;
        let k_dst = k_buf.get_mut(byte_range.clone()).ok_or_else(buffer_err)?;
        let v_dst = v_buf.get_mut(byte_range).ok_or_else(buffer_err)?;
        let slot = self.lens.get_mut(layer_idx).ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers,
            }
            .build()
        })?;
        k_dst.copy_from_slice(k_bytes);
        v_dst.copy_from_slice(v_bytes);
        *slot = next_len;
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
                num_layers: self.layout.num_layers(),
            }
            .build()
        })?;
        let geometry = self.append_geometry(layer_idx, current, n_k)?;
        self.write_append(layer_idx, geometry, &k_bytes, &v_bytes)
    }

    fn get(&self, layer_idx: usize, len: usize) -> Result<(Tensor, Tensor)> {
        self.check_layer(layer_idx)?;
        let current = self.lens.get(layer_idx).copied().ok_or_else(|| {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers: self.layout.num_layers(),
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
            GeometryOverflowSnafu {
                op: "FlatKvCache::get",
                msg: format!("len {len} × row_bytes {row_bytes} overflows"),
            }
            .build()
        })?;
        let layer_err = || {
            LayerOutOfRangeSnafu {
                layer_idx,
                num_layers: self.layout.num_layers(),
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
        let k = cpu_tensor_from_bytes(self.layout.dtype(), k_slice, shape.clone())?;
        let v = cpu_tensor_from_bytes(self.layout.dtype(), v_slice, shape)?;
        Ok((k, v))
    }

    fn len_of(&self, layer_idx: usize) -> Option<usize> {
        self.lens.get(layer_idx).copied()
    }

    fn num_layers(&self) -> usize {
        self.layout.num_layers()
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
/// its little-endian encoding, so `_to_le` goes unused by construction.
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
fn le_bytes_of<T: Copy, const N: usize>(
    v: &[T],
    to_le: impl Fn(T) -> [u8; N],
) -> Result<Cow<'_, [u8]>> {
    let byte_len = v.len().checked_mul(N).ok_or_else(|| {
        GeometryOverflowSnafu {
            op: "le_bytes_of",
            msg: format!("{} elements × {N} bytes overflows", v.len()),
        }
        .build()
    })?;
    let mut out = try_reserve_vec(byte_len, "big-endian marshal buffer")?;
    for &x in v {
        out.extend_from_slice(&to_le(x));
    }
    Ok(Cow::Owned(out))
}

fn cpu_storage_bytes(s: &CpuStorage) -> Result<Cow<'_, [u8]>> {
    let bytes = match s {
        CpuStorage::F32(v) => le_bytes_of(v, f32::to_le_bytes),
        CpuStorage::F16(v) => le_bytes_of(v, half::f16::to_le_bytes),
        CpuStorage::BF16(v) => le_bytes_of(v, half::bf16::to_le_bytes),
        CpuStorage::I32(v) => le_bytes_of(v, i32::to_le_bytes),
        CpuStorage::I8(v) => le_bytes_of(v, i8::to_le_bytes),
        CpuStorage::U8(v) => return Ok(Cow::Borrowed(v.as_slice())),
        _ => {
            return UnsupportedStorageSnafu {
                msg: "unsupported future CpuStorage variant",
            }
            .fail();
        }
    };
    #[cfg(target_endian = "little")]
    {
        Ok(bytes)
    }
    #[cfg(not(target_endian = "little"))]
    {
        bytes
    }
}

fn allocate_zeroed(len: usize, what: &'static str) -> Result<Vec<u8>> {
    let mut buffer = try_reserve_vec(len, what)?;
    buffer.resize(len, 0);
    Ok(buffer)
}

fn try_reserve_vec<T>(requested_elements: usize, what: &'static str) -> Result<Vec<T>> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(requested_elements)
        .context(AllocationSnafu {
            what,
            requested_elements,
        })?;
    Ok(buffer)
}

fn cpu_tensor_from_bytes(dtype: DType, bytes: &[u8], shape: Shape) -> Result<Tensor> {
    let elem_count = shape.checked_elem_count()?;
    let storage = match dtype {
        DType::F32 => CpuStorage::F32(chunks_to_f32(bytes, elem_count)?),
        DType::F16 => CpuStorage::F16(chunks_to_f16(bytes, elem_count)?),
        DType::BF16 => CpuStorage::BF16(chunks_to_bf16(bytes, elem_count)?),
        DType::I32 => CpuStorage::I32(chunks_to_i32(bytes, elem_count)?),
        DType::I8 => CpuStorage::I8(bytes_to_i8(bytes)?),
        DType::U8 => CpuStorage::U8(copy_bytes(bytes, "decoded U8 storage")?),
        other => {
            return MsgSnafu {
                message: format!("dtype {other:?} not supported by Phase-2 FlatKvCache"),
            }
            .fail();
        }
    };
    Ok(Tensor::try_from_cpu(storage, shape)?)
}

/// Postcondition on every `chunks_exact`-based decoder: `chunks_exact`
/// silently drops a trailing partial chunk, so a byte length that isn't
/// a whole multiple of the dtype width would otherwise produce a `Vec`
/// shorter than `expected`. Checked tensor construction independently rejects
/// malformed decoding, so it cannot cross either boundary.
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

fn copy_bytes(bytes: &[u8], what: &'static str) -> Result<Vec<u8>> {
    let mut out = try_reserve_vec(bytes.len(), what)?;
    out.extend_from_slice(bytes);
    Ok(out)
}

fn bytes_to_i8(bytes: &[u8]) -> Result<Vec<i8>> {
    let mut out = try_reserve_vec(bytes.len(), "decoded I8 storage")?;
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
    Ok(out)
}

fn chunks_to_f32(bytes: &[u8], elem: usize) -> Result<Vec<f32>> {
    let mut out = try_reserve_vec(elem, "decoded F32 storage")?;
    for c in bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(c);
        out.push(f32::from_le_bytes(b));
    }
    check_decoded_len(out.len(), elem, bytes.len())?;
    Ok(out)
}
fn chunks_to_i32(bytes: &[u8], elem: usize) -> Result<Vec<i32>> {
    let mut out = try_reserve_vec(elem, "decoded I32 storage")?;
    for c in bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(c);
        out.push(i32::from_le_bytes(b));
    }
    check_decoded_len(out.len(), elem, bytes.len())?;
    Ok(out)
}
fn chunks_to_f16(bytes: &[u8], elem: usize) -> Result<Vec<half::f16>> {
    let mut out = try_reserve_vec(elem, "decoded F16 storage")?;
    for c in bytes.chunks_exact(2) {
        let mut b = [0u8; 2];
        b.copy_from_slice(c);
        out.push(half::f16::from_le_bytes(b));
    }
    check_decoded_len(out.len(), elem, bytes.len())?;
    Ok(out)
}
fn chunks_to_bf16(bytes: &[u8], elem: usize) -> Result<Vec<half::bf16>> {
    let mut out = try_reserve_vec(elem, "decoded BF16 storage")?;
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
