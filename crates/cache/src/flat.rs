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
    bytes: &'t [u8],
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
        let storage = t.cpu_storage().ok_or_else(|| Error::ShapeMismatch {
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
            .copy_from_slice(k_bytes);
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
            .copy_from_slice(v_bytes);
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

    fn len_of(&self, layer_idx: usize) -> usize {
        self.lens.get(layer_idx).copied().unwrap_or(0)
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

fn cpu_storage_bytes(s: &CpuStorage) -> Result<&[u8]> {
    // SAFETY: each variant holds a `Vec<T>` of a `BytePod`-compatible
    // `T`. Reinterpreting as a byte slice is defined because `T` has
    // `Copy` + every bit pattern is valid (f32/f16/bf16/i*/u8).
    unsafe {
        match s {
            CpuStorage::F32(v) => Ok(core::slice::from_raw_parts(
                v.as_ptr().cast::<u8>(),
                core::mem::size_of_val(v.as_slice()),
            )),
            CpuStorage::F16(v) => Ok(core::slice::from_raw_parts(
                v.as_ptr().cast::<u8>(),
                core::mem::size_of_val(v.as_slice()),
            )),
            CpuStorage::BF16(v) => Ok(core::slice::from_raw_parts(
                v.as_ptr().cast::<u8>(),
                core::mem::size_of_val(v.as_slice()),
            )),
            CpuStorage::I32(v) => Ok(core::slice::from_raw_parts(
                v.as_ptr().cast::<u8>(),
                core::mem::size_of_val(v.as_slice()),
            )),
            CpuStorage::I8(v) => Ok(core::slice::from_raw_parts(
                v.as_ptr().cast::<u8>(),
                core::mem::size_of_val(v.as_slice()),
            )),
            CpuStorage::U8(v) => Ok(v.as_slice()),
            _ => Err(Error::ShapeMismatch {
                msg: "unsupported future CpuStorage variant".into(),
            }),
        }
    }
}

fn cpu_tensor_from_bytes(dtype: DType, bytes: &[u8], shape: Shape) -> Result<Tensor> {
    let elem_count = shape.elem_count();
    let storage = match dtype {
        DType::F32 => CpuStorage::F32(chunks_to_f32(bytes, elem_count)),
        DType::F16 => CpuStorage::F16(chunks_to_f16(bytes, elem_count)),
        DType::BF16 => CpuStorage::BF16(chunks_to_bf16(bytes, elem_count)),
        DType::I32 => CpuStorage::I32(chunks_to_i32(bytes, elem_count)),
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

fn chunks_to_f32(bytes: &[u8], elem: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(c);
        out.push(f32::from_le_bytes(b));
    }
    out
}
fn chunks_to_i32(bytes: &[u8], elem: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(c);
        out.push(i32::from_le_bytes(b));
    }
    out
}
fn chunks_to_f16(bytes: &[u8], elem: usize) -> Vec<half::f16> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(2) {
        let mut b = [0u8; 2];
        b.copy_from_slice(c);
        out.push(half::f16::from_le_bytes(b));
    }
    out
}
fn chunks_to_bf16(bytes: &[u8], elem: usize) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(elem);
    for c in bytes.chunks_exact(2) {
        let mut b = [0u8; 2];
        b.copy_from_slice(c);
        out.push(half::bf16::from_le_bytes(b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_small() -> CacheLayout {
        CacheLayout {
            num_layers: 4,
            num_kv_heads: 2,
            head_dim: 3,
            max_seq_len: 8,
            dtype: DType::F32,
        }
    }

    fn one_row_tensor(val: f32, layout: &CacheLayout) -> Tensor {
        let row = vec![val; layout.row_elems()];
        Tensor::from_cpu(CpuStorage::F32(row), Shape::new(&[1, layout.row_elems()]))
    }

    fn host_f32(t: &Tensor) -> Vec<f32> {
        match t.cpu_storage() {
            Some(CpuStorage::F32(v)) => v.clone(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn put_then_get_round_trip() -> Result<()> {
        let layout = layout_small();
        let mut c = FlatKvCache::new(layout);
        let k = one_row_tensor(1.5, &layout);
        let v = one_row_tensor(2.5, &layout);
        c.put(0, &k, &v)?;
        assert_eq!(c.len_of(0), 1);
        let (k_out, v_out) = c.get(0, 1)?;
        assert_eq!(k_out.dims(), &[1, layout.row_elems()]);
        let k_host = host_f32(&k_out);
        assert_eq!(k_host.len(), layout.row_elems());
        for x in &k_host {
            assert!((x - 1.5).abs() < 1e-6);
        }
        let v_host = host_f32(&v_out);
        for x in &v_host {
            assert!((x - 2.5).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn put_then_get_round_trip_f16() -> Result<()> {
        // WHY(forkwright/logismos#42): every prior round-trip test used
        // only DType::F32. F16 goes through `chunks_to_f16` on read and
        // a native-endian raw-byte reinterpret on write — a path never
        // exercised before this test.
        let layout = CacheLayout {
            dtype: DType::F16,
            ..layout_small()
        };
        let mut c = FlatKvCache::new(layout);
        let val = half::f16::from_f32(1.5);
        let row = vec![val; layout.row_elems()];
        let k = Tensor::from_cpu(
            CpuStorage::F16(row.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        let v = Tensor::from_cpu(CpuStorage::F16(row), Shape::new(&[1, layout.row_elems()]));
        c.put(0, &k, &v)?;
        let (k_out, v_out) = c.get(0, 1)?;
        let Some(CpuStorage::F16(k_host)) = k_out.cpu_storage() else {
            return Err(Error::Msg("expected F16 storage".into()));
        };
        assert_eq!(k_host, &vec![val; layout.row_elems()]);
        let Some(CpuStorage::F16(v_host)) = v_out.cpu_storage() else {
            return Err(Error::Msg("expected F16 storage".into()));
        };
        assert_eq!(v_host, &vec![val; layout.row_elems()]);
        Ok(())
    }

    #[test]
    fn put_then_get_round_trip_bf16() -> Result<()> {
        let layout = CacheLayout {
            dtype: DType::BF16,
            ..layout_small()
        };
        let mut c = FlatKvCache::new(layout);
        let val = half::bf16::from_f32(-2.25);
        let row = vec![val; layout.row_elems()];
        let k = Tensor::from_cpu(
            CpuStorage::BF16(row.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        let v = Tensor::from_cpu(CpuStorage::BF16(row), Shape::new(&[1, layout.row_elems()]));
        c.put(0, &k, &v)?;
        let (k_out, v_out) = c.get(0, 1)?;
        let Some(CpuStorage::BF16(k_host)) = k_out.cpu_storage() else {
            return Err(Error::Msg("expected BF16 storage".into()));
        };
        assert_eq!(k_host, &vec![val; layout.row_elems()]);
        let Some(CpuStorage::BF16(v_host)) = v_out.cpu_storage() else {
            return Err(Error::Msg("expected BF16 storage".into()));
        };
        assert_eq!(v_host, &vec![val; layout.row_elems()]);
        Ok(())
    }

    #[test]
    fn put_then_get_round_trip_i32() -> Result<()> {
        let layout = CacheLayout {
            dtype: DType::I32,
            ..layout_small()
        };
        let mut c = FlatKvCache::new(layout);
        let row_k = vec![7_i32; layout.row_elems()];
        let row_v = vec![-3_i32; layout.row_elems()];
        let k = Tensor::from_cpu(
            CpuStorage::I32(row_k.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        let v = Tensor::from_cpu(
            CpuStorage::I32(row_v.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        c.put(0, &k, &v)?;
        let (k_out, v_out) = c.get(0, 1)?;
        let Some(CpuStorage::I32(k_host)) = k_out.cpu_storage() else {
            return Err(Error::Msg("expected I32 storage".into()));
        };
        assert_eq!(k_host, &row_k);
        let Some(CpuStorage::I32(v_host)) = v_out.cpu_storage() else {
            return Err(Error::Msg("expected I32 storage".into()));
        };
        assert_eq!(v_host, &row_v);
        Ok(())
    }

    #[test]
    fn put_then_get_round_trip_i8() -> Result<()> {
        // WHY(forkwright/logismos#42): this is the dtype whose reader
        // used `from_ne_bytes` instead of the little-endian convention
        // every other reader uses — a quantized (I8) on-device model is
        // exactly the path this cache exists to serve.
        let layout = CacheLayout {
            dtype: DType::I8,
            ..layout_small()
        };
        let mut c = FlatKvCache::new(layout);
        let row_k = vec![i8::MIN; layout.row_elems()];
        let row_v = vec![i8::MAX; layout.row_elems()];
        let k = Tensor::from_cpu(
            CpuStorage::I8(row_k.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        let v = Tensor::from_cpu(
            CpuStorage::I8(row_v.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        c.put(0, &k, &v)?;
        let (k_out, v_out) = c.get(0, 1)?;
        let Some(CpuStorage::I8(k_host)) = k_out.cpu_storage() else {
            return Err(Error::Msg("expected I8 storage".into()));
        };
        assert_eq!(k_host, &row_k);
        let Some(CpuStorage::I8(v_host)) = v_out.cpu_storage() else {
            return Err(Error::Msg("expected I8 storage".into()));
        };
        assert_eq!(v_host, &row_v);
        Ok(())
    }

    #[test]
    fn put_then_get_round_trip_u8() -> Result<()> {
        let layout = CacheLayout {
            dtype: DType::U8,
            ..layout_small()
        };
        let mut c = FlatKvCache::new(layout);
        let row_k = vec![200_u8; layout.row_elems()];
        let row_v = vec![1_u8; layout.row_elems()];
        let k = Tensor::from_cpu(
            CpuStorage::U8(row_k.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        let v = Tensor::from_cpu(
            CpuStorage::U8(row_v.clone()),
            Shape::new(&[1, layout.row_elems()]),
        );
        c.put(0, &k, &v)?;
        let (k_out, v_out) = c.get(0, 1)?;
        let Some(CpuStorage::U8(k_host)) = k_out.cpu_storage() else {
            return Err(Error::Msg("expected U8 storage".into()));
        };
        assert_eq!(k_host, &row_k);
        let Some(CpuStorage::U8(v_host)) = v_out.cpu_storage() else {
            return Err(Error::Msg("expected U8 storage".into()));
        };
        assert_eq!(v_host, &row_v);
        Ok(())
    }

    #[test]
    fn grows_monotonically_across_layers() -> Result<()> {
        let layout = layout_small();
        let mut c = FlatKvCache::new(layout);
        for _ in 0..3 {
            for layer in 0..layout.num_layers {
                let k = one_row_tensor(0.1, &layout);
                let v = one_row_tensor(0.2, &layout);
                c.put(layer, &k, &v)?;
            }
        }
        for layer in 0..layout.num_layers {
            assert_eq!(c.len_of(layer), 3);
        }
        Ok(())
    }

    #[test]
    fn reset_zeros_lengths() -> Result<()> {
        let layout = layout_small();
        let mut c = FlatKvCache::new(layout);
        let k = one_row_tensor(1.0, &layout);
        let v = one_row_tensor(1.0, &layout);
        c.put(0, &k, &v)?;
        c.put(1, &k, &v)?;
        c.reset();
        assert_eq!(c.len_of(0), 0);
        assert_eq!(c.len_of(1), 0);
        // Re-write after reset stays consistent.
        c.put(0, &k, &v)?;
        assert_eq!(c.len_of(0), 1);
        Ok(())
    }

    #[test]
    fn overflow_errors_cleanly() -> Result<()> {
        let layout = CacheLayout {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            max_seq_len: 2,
            dtype: DType::F32,
        };
        let mut c = FlatKvCache::new(layout);
        let k = one_row_tensor(1.0, &layout);
        let v = one_row_tensor(1.0, &layout);
        c.put(0, &k, &v)?;
        c.put(0, &k, &v)?;
        let err = c.put(0, &k, &v);
        assert!(matches!(err, Err(Error::LenOverflow { .. })));
        Ok(())
    }

    #[test]
    fn read_beyond_written_errors() {
        let layout = layout_small();
        let c = FlatKvCache::new(layout);
        let err = c.get(0, 1);
        assert!(matches!(err, Err(Error::ReadBeyondWritten { .. })));
    }

    #[test]
    fn layer_out_of_range_errors() {
        let layout = layout_small();
        let c = FlatKvCache::new(layout);
        let err = c.get(99, 0);
        assert!(matches!(err, Err(Error::LayerOutOfRange { .. })));
    }

    #[test]
    fn buffer_bytes_computed_correctly() {
        let layout = layout_small();
        // dtype F32 (4B) × 2 heads × 3 head_dim × 8 max_seq = 192 B
        assert_eq!(layout.buffer_bytes(), 192);
        assert_eq!(layout.row_elems(), 6);
        assert_eq!(layout.row_bytes(), 24);
    }
}
