//! `gguf.rs` tests, `#[path]`-included as a sibling file per `RUST/file-too-long`.

use super::*;

/// Build a minimal v3 GGUF file in memory with:
/// - magic + version
/// - one metadata KV: "answer" = u32(42)
/// - one tensor: "one", shape=[3], F32, bytes = [1.0, 2.0, 3.0]
///
/// `pub(crate)` so `lib.rs`'s `Archive::open` dispatch test can
/// reuse it rather than duplicating a fixture builder.
pub(crate) fn fixture_bytes() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    // tensor count, metadata count
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    // metadata kv 1: "answer" u32 42
    let key = "answer";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes()); // type = U32
    buf.extend_from_slice(&42u32.to_le_bytes());
    // metadata kv 2: "general.alignment" u32 32 (default, but
    // explicit so the test doubles as an alignment parse test)
    let key = "general.alignment";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&32u32.to_le_bytes());
    // tensor 1: "one" F32 [3] offset 0
    let tname = "one";
    buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
    buf.extend_from_slice(tname.as_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&3u64.to_le_bytes()); // dim 0
    buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // data offset
    // align to 32
    let pad = align_up(buf.len() as u64, 32) as usize - buf.len();
    buf.extend(std::iter::repeat_n(0u8, pad));
    // payload: [1.0, 2.0, 3.0] as f32 LE
    for v in [1.0_f32, 2.0, 3.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

#[test]
fn align_up_rounds_correctly() {
    assert_eq!(align_up(0, 32), 0);
    assert_eq!(align_up(1, 32), 32);
    assert_eq!(align_up(32, 32), 32);
    assert_eq!(align_up(33, 32), 64);
}

#[test]
fn reads_fixture_bytes() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("fixture.gguf");
    std::fs::write(&path, fixture_bytes())?;

    let r = Reader::open(&path)?;
    assert_eq!(r.len(), 1);
    assert_eq!(r.names(), vec!["one"]);
    assert!(matches!(
        r.metadata().get("answer"),
        Some(MetaValue::U32(42))
    ));
    let tv = r.get("one")?;
    assert_eq!(tv.dtype, taxis::DType::F32);
    assert_eq!(tv.shape, vec![3]);
    // WHY(forkwright/logismos#56): a length-only assertion cannot pin
    // `byte_range_for`'s `checked_add` rewrite — an off-by-N in
    // `start`/`end` still yields a 12-byte slice of the correct
    // dtype/shape, just the wrong bytes. Assert the decoded content.
    let expected: Vec<u8> = [1.0_f32, 2.0, 3.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(tv.bytes, expected.as_slice());
    Ok(())
}

#[test]
fn array_metadata_type_parses_elements() -> Result<()> {
    // WHY(forkwright/logismos#37): the array branch (type id 9) of
    // `read_meta_value_typed` had no test coverage at all — this
    // exercises length read, inner_type read, allocation, and the
    // per-element recursive decode.
    let mut buf = Vec::new();
    buf.extend_from_slice(&4u32.to_le_bytes()); // inner_type = U32
    buf.extend_from_slice(&3u64.to_le_bytes()); // n = 3 elements
    for v in [10u32, 20, 30] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut cur = Cursor::new(&buf);
    let value = cur.read_meta_value_typed(9)?;
    let MetaValue::Array(items) = value else {
        return Err(Error::Gguf {
            offset: 0,
            msg: "expected MetaValue::Array".into(),
        });
    };
    assert_eq!(items.len(), 3);
    assert!(matches!(items[0], MetaValue::U32(10)));
    assert!(matches!(items[1], MetaValue::U32(20)));
    assert!(matches!(items[2], MetaValue::U32(30)));
    Ok(())
}

#[test]
fn nested_array_inner_type_is_rejected() {
    // WHY(forkwright/logismos#35): the GGUF v3 spec forbids
    // arrays-of-arrays. Before the fix, `inner_type = 9` recursed
    // unboundedly — a crafted file chaining this at every nesting
    // level exhausts the thread stack (an unrecoverable SIGSEGV, not
    // a catchable error). The very first nested level must instead
    // return `Err`.
    let inner_type_bytes = 9u32.to_le_bytes();
    let mut cur = Cursor::new(&inner_type_bytes);
    let result = cur.read_meta_value_typed(9);
    assert!(matches!(result, Err(Error::Gguf { .. })));
}

#[test]
fn huge_tensor_count_returns_err_not_abort() -> Result<()> {
    // WHY(forkwright/logismos#34): before the fix this pre-allocated
    // `Vec::with_capacity(tensor_count_usize)` for an untrusted,
    // attacker-controlled count straight off the wire — enough to
    // abort the process through the allocator rather than returning
    // a `Result::Err` a caller could handle. A file that claims an
    // enormous tensor count but has no data behind it must fail fast
    // with a returned error instead.
    let dir = tempdir_for_test();
    let path = dir.join("huge-tensor-count.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    std::fs::write(&path, buf)?;

    let result = Reader::open(&path);
    assert!(matches!(result, Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn dims_product_overflow_is_rejected_not_wrapped_to_zero() -> Result<()> {
    // WHY(forkwright/logismos#36): `desc.dims.iter().product()` used
    // ordinary wrapping multiplication in a release build. A dims
    // vector whose product exceeds `u64::MAX` could silently wrap to
    // `0`, producing a zero-byte `TensorView` that still reports the
    // original (enormous) `dims` in its `shape` — passing every
    // bounds check unchanged. The checked-multiply loop must instead
    // return `Err`.
    let dir = tempdir_for_test();
    let path = dir.join("dims-overflow.gguf");
    std::fs::write(&path, fixture_bytes())?;
    let r = Reader::open(&path)?;

    let desc = TensorDescriptor {
        name: "overflow".to_string(),
        dims: vec![u64::MAX, 2],
        ggml_type: GgmlType::F32,
        data_offset: 0,
    };
    let result = r.byte_range_for(&desc);
    assert!(matches!(result, Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn byte_range_for_rejects_byte_count_overflow() -> Result<()> {
    // WHY(forkwright/logismos#56): distinct from the dims-product
    // overflow above — this exercises the *new* `checked_mul` on
    // `bits * elem_count` (the byte-count multiply), which used to
    // be `saturating_mul` and clamp instead of erroring.
    let dir = tempdir_for_test();
    let path = dir.join("byte-count-overflow.gguf");
    std::fs::write(&path, fixture_bytes())?;
    let r = Reader::open(&path)?;

    let desc = TensorDescriptor {
        name: "huge".to_string(),
        // A single dim of `u64::MAX` doesn't overflow the
        // dims-product loop (1 * u64::MAX doesn't overflow), so it
        // reaches the byte-count multiply as `elem_count =
        // usize::MAX`, which `32 bits * usize::MAX` then overflows.
        dims: vec![u64::MAX],
        ggml_type: GgmlType::F32,
        data_offset: 0,
    };
    let result = r.byte_range_for(&desc);
    // WHY(forkwright/logismos#56, redden-verified): a bare
    // `matches!(.., Err(Error::Gguf { .. }))` here would also pass
    // against a REVERTED `saturating_mul` — the clamped byte count is
    // so large it still trips the downstream `end_usize >
    // self.mmap.len()` bounds check with a *different* message, so the
    // outer error variant alone can't tell "rejected at the multiply"
    // from "rejected later for an unrelated reason". Confirmed by
    // actually reverting checked_mul on a throwaway branch and running
    // this exact assertion: it passed unchanged (CI run 31924612522,
    // job 95110102176, `byte_range_for_rejects_byte_count_overflow`
    // logged PASS). Asserting the message distinguishes them: only the
    // checked_mul path says "overflows usize".
    let Err(Error::Gguf { msg, .. }) = &result else {
        return Err(Error::Gguf {
            offset: 0,
            msg: format!("expected Error::Gguf, got {result:?}"),
        });
    };
    assert!(
        msg.contains("overflows usize"),
        "expected the checked_mul overflow message, got: {msg}"
    );
    Ok(())
}

#[test]
fn byte_range_for_rejects_out_of_bounds_data() -> Result<()> {
    // WHY(forkwright/logismos#56): `byte_range_for`'s
    // `end_usize > mmap.len()` branch had no test — every existing
    // fixture's tensor data fits inside the file.
    let dir = tempdir_for_test();
    let path = dir.join("oob-tensor-data.gguf");
    std::fs::write(&path, fixture_bytes())?;
    let r = Reader::open(&path)?;

    let desc = TensorDescriptor {
        name: "past-eof".to_string(),
        dims: vec![1_000_000],
        ggml_type: GgmlType::F32,
        data_offset: 0,
    };
    let result = r.byte_range_for(&desc);
    assert!(matches!(result, Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn rejects_unsupported_gguf_version() -> Result<()> {
    // WHY(forkwright/logismos#56): the `version != GGUF_V3` branch
    // had no test at all.
    let dir = tempdir_for_test();
    let path = dir.join("bad-version.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&99u32.to_le_bytes()); // unsupported version
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    std::fs::write(&path, buf)?;

    let result = Reader::open(&path);
    assert!(matches!(result, Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn read_string_rejects_invalid_utf8() {
    // WHY(forkwright/logismos#56): `read_string`'s invalid-UTF-8
    // rejection path had no test.
    let mut buf = Vec::new();
    buf.extend_from_slice(&2u64.to_le_bytes()); // declared length = 2
    buf.extend_from_slice(&[0xFF, 0xFE]); // not valid UTF-8
    let mut cur = Cursor::new(&buf);
    let result = cur.read_string();
    assert!(matches!(result, Err(Error::Gguf { .. })));
}

#[test]
fn rejects_zero_length_dimension() -> Result<()> {
    // WHY(forkwright/logismos#60): a `0` entry in `dims` used to
    // pass validation and produce a zero-byte tensor that still
    // reports the original shape; it must now be rejected at parse
    // time.
    let dir = tempdir_for_test();
    let path = dir.join("zero-dim.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    let tname = "zero";
    buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
    buf.extend_from_slice(tname.as_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    buf.extend_from_slice(&3u64.to_le_bytes()); // dim 0 = 3
    buf.extend_from_slice(&0u64.to_le_bytes()); // dim 1 = 0 (offending)
    buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // data offset
    std::fs::write(&path, buf)?;

    let result = Reader::open(&path);
    assert!(matches!(result, Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn rejects_duplicate_metadata_key() -> Result<()> {
    // WHY(forkwright/logismos#60): two KV entries sharing a key used
    // to silently overwrite; the second one must now fail instead.
    let dir = tempdir_for_test();
    let path = dir.join("dup-metadata-key.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&2u64.to_le_bytes()); // metadata count
    for _ in 0..2 {
        let key = "dup";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // type = U32
        buf.extend_from_slice(&1u32.to_le_bytes());
    }
    std::fs::write(&path, buf)?;

    let result = Reader::open(&path);
    assert!(matches!(result, Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn rejects_duplicate_tensor_name() -> Result<()> {
    // WHY(forkwright/logismos#60): two tensor descriptors sharing a
    // name used to silently overwrite `tensor_by_name`; the second
    // one must now fail instead.
    let dir = tempdir_for_test();
    let path = dir.join("dup-tensor-name.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    for _ in 0..2 {
        let tname = "dup";
        buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        buf.extend_from_slice(tname.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&1u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // data offset
    }
    std::fs::write(&path, buf)?;

    let result = Reader::open(&path);
    assert!(matches!(result, Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn get_rejects_after_external_truncation() -> Result<()> {
    // WHY(forkwright/logismos#60): the mmap SAFETY comment states
    // the backing file must not be mutated while the mapping is
    // open; nothing enforced that. `get` now re-stats the file and
    // refuses a mapping whose backing file has changed size since
    // it was opened.
    let dir = tempdir_for_test();
    let path = dir.join("truncated-after-open.gguf");
    std::fs::write(&path, fixture_bytes())?;
    let r = Reader::open(&path)?;
    std::fs::write(&path, b"short")?;

    let result = r.get("one");
    assert!(matches!(result, Err(Error::MmapStale { .. })));
    Ok(())
}

fn tempdir_for_test() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("logismos-gguf-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&p);
    p
}
