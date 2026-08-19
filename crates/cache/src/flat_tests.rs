use super::*;
// WHY not via `use super::*`: `flat.rs` no longer imports `Error` (nothing
// outside these tests references it), so the assertions below import it
// directly.
use crate::error::Error;

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
    assert_eq!(c.len_of(0), Some(1));
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
        return MsgSnafu {
            message: "expected F16 storage",
        }
        .fail();
    };
    assert_eq!(k_host, &vec![val; layout.row_elems()]);
    let Some(CpuStorage::F16(v_host)) = v_out.cpu_storage() else {
        return MsgSnafu {
            message: "expected F16 storage",
        }
        .fail();
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
        return MsgSnafu {
            message: "expected BF16 storage",
        }
        .fail();
    };
    assert_eq!(k_host, &vec![val; layout.row_elems()]);
    let Some(CpuStorage::BF16(v_host)) = v_out.cpu_storage() else {
        return MsgSnafu {
            message: "expected BF16 storage",
        }
        .fail();
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
        return MsgSnafu {
            message: "expected I32 storage",
        }
        .fail();
    };
    assert_eq!(k_host, &row_k);
    let Some(CpuStorage::I32(v_host)) = v_out.cpu_storage() else {
        return MsgSnafu {
            message: "expected I32 storage",
        }
        .fail();
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
        return MsgSnafu {
            message: "expected I8 storage",
        }
        .fail();
    };
    assert_eq!(k_host, &row_k);
    let Some(CpuStorage::I8(v_host)) = v_out.cpu_storage() else {
        return MsgSnafu {
            message: "expected I8 storage",
        }
        .fail();
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
        return MsgSnafu {
            message: "expected U8 storage",
        }
        .fail();
    };
    assert_eq!(k_host, &row_k);
    let Some(CpuStorage::U8(v_host)) = v_out.cpu_storage() else {
        return MsgSnafu {
            message: "expected U8 storage",
        }
        .fail();
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
        assert_eq!(c.len_of(layer), Some(3));
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
    assert_eq!(c.len_of(0), Some(0));
    assert_eq!(c.len_of(1), Some(0));
    // Re-write after reset stays consistent.
    c.put(0, &k, &v)?;
    assert_eq!(c.len_of(0), Some(1));
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
#[expect(
    clippy::cast_precision_loss,
    reason = "usize test indices (n_tokens * row_elems) stay single-digit here, far below f32's 24-bit exact-integer bound, so the cast is lossless"
)]
fn put_multi_token_batch_verifies_values() -> Result<()> {
    // WHY(forkwright/logismos-archive#70): every prior put/get test
    // writes exactly one row per call — including the multi-call
    // `grows_monotonically_across_layers` test, which still puts
    // n_tokens=1 per call. A single `put` with n_tokens > 1 exercises
    // a different path: one contiguous multi-row `tensor_as_bytes`
    // slice copied in one `copy_from_slice`, not several single-row
    // writes landing at the same offset.
    let layout = layout_small();
    let mut c = FlatKvCache::new(layout);
    let row = layout.row_elems();
    let n = 3;
    let k_data: Vec<f32> = (0..n * row).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..n * row).map(|i| -(i as f32) - 1.0).collect();
    let k = Tensor::from_cpu(CpuStorage::F32(k_data.clone()), Shape::new(&[n, row]));
    let v = Tensor::from_cpu(CpuStorage::F32(v_data.clone()), Shape::new(&[n, row]));
    c.put(0, &k, &v)?;
    assert_eq!(c.len_of(0), Some(n));
    let (k_out, v_out) = c.get(0, n)?;
    assert_eq!(k_out.dims(), &[n, row]);
    assert_eq!(host_f32(&k_out), k_data);
    assert_eq!(host_f32(&v_out), v_data);
    Ok(())
}

#[test]
fn get_zero_len_returns_empty_tensor() -> Result<()> {
    // WHY(forkwright/logismos-archive#70): the existing `get`-path
    // tests cover past-written-length (`read_beyond_written_errors`)
    // and out-of-range layer (`layer_out_of_range_errors`); neither
    // exercises the in-range success path of asking for zero rows.
    let layout = layout_small();
    let c = FlatKvCache::new(layout);
    let (k, v) = c.get(0, 0)?;
    assert_eq!(k.dims(), &[0, layout.row_elems()]);
    assert_eq!(v.dims(), &[0, layout.row_elems()]);
    assert!(host_f32(&k).is_empty());
    assert!(host_f32(&v).is_empty());
    Ok(())
}

#[test]
fn len_of_distinguishes_out_of_range_from_unwritten() {
    // WHY(forkwright/logismos-archive#70): the pre-fix `len_of`
    // returned plain `0` for both an unwritten in-range layer and an
    // out-of-range one (`.unwrap_or(0)`), indistinguishable from the
    // caller's side. `Some(0)` vs `None` makes the two cases distinct
    // at the type level.
    let layout = layout_small();
    let c = FlatKvCache::new(layout);
    assert_eq!(c.len_of(0), Some(0));
    assert_eq!(c.len_of(layout.num_layers), None);
    assert_eq!(c.len_of(9999), None);
}

#[test]
fn chunks_to_f32_rejects_undersized_trailing_bytes() {
    // WHY(forkwright/logismos-archive#70): `chunks_exact(4)` silently
    // drops a trailing partial chunk. 6 bytes decodes to 1 f32 (4
    // bytes consumed, 2 dropped) — if the caller expected 2 elements
    // (e.g. a `Shape` claiming 2 rows), the pre-fix code returned a
    // 1-element `Vec` with no error, and the resulting `Tensor`
    // silently disagreed with its own declared shape.
    let bytes = [0u8; 6];
    let err = chunks_to_f32(&bytes, 2);
    assert!(matches!(err, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn cpu_storage_bytes_pins_little_endian_encoding() -> Result<()> {
    // WHY(forkwright/logismos-archive#70): a round-trip through this
    // crate's own encoder (`cpu_storage_bytes`) and decoder
    // (`chunks_to_*`) pair passes even when both sides agree on the
    // same WRONG byte order — which is exactly what shipped before
    // this fix: `cpu_storage_bytes` reinterpreted native-endian
    // bytes while the decoders always read little-endian, so the two
    // agreed only on a little-endian host. Pinning against
    // externally-known IEEE-754 / two's-complement little-endian
    // encodings (verified independently of this crate) catches what
    // a round-trip cannot.
    let f32_storage = CpuStorage::F32(vec![1.0f32]);
    let f32_bytes = cpu_storage_bytes(&f32_storage)?;
    assert_eq!(f32_bytes.to_vec(), vec![0x00u8, 0x00, 0x80, 0x3F]);

    let i32_storage = CpuStorage::I32(vec![0x0102_0304]);
    let i32_bytes = cpu_storage_bytes(&i32_storage)?;
    assert_eq!(i32_bytes.to_vec(), vec![0x04u8, 0x03, 0x02, 0x01]);

    let f16_storage = CpuStorage::F16(vec![half::f16::from_f32(1.0)]);
    let f16_bytes = cpu_storage_bytes(&f16_storage)?;
    assert_eq!(f16_bytes.to_vec(), vec![0x00u8, 0x3C]);

    let bf16_storage = CpuStorage::BF16(vec![half::bf16::from_f32(1.0)]);
    let bf16_bytes = cpu_storage_bytes(&bf16_storage)?;
    assert_eq!(bf16_bytes.to_vec(), vec![0x80u8, 0x3F]);

    Ok(())
}

#[test]
fn buffer_bytes_computed_correctly() {
    let layout = layout_small();
    // dtype F32 (4B) × 2 heads × 3 head_dim × 8 max_seq = 192 B
    assert_eq!(layout.buffer_bytes(), 192);
    assert_eq!(layout.row_elems(), 6);
    assert_eq!(layout.row_bytes(), 24);
}
