//! `gguf.rs` tests, `#[path]`-included as a sibling file per `RUST/file-too-long`.

use super::*;
// WHY not via `use super::*`: the parent's `Error` import is declared
// `#[expect(unused_imports)]` for intra-doc links; resolving these
// assertions through this dedicated import keeps that expectation
// fulfilled in test builds.
use crate::error::Error;

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
    let pad = (32 - buf.len() % 32) % 32;
    buf.extend(std::iter::repeat_n(0u8, pad));
    // payload: [1.0, 2.0, 3.0] as f32 LE
    for v in [1.0_f32, 2.0, 3.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

#[test]
fn align_up_rounds_correctly() -> Result<()> {
    assert_eq!(align_up(0, 32)?, 0);
    assert_eq!(align_up(1, 32)?, 32);
    assert_eq!(align_up(32, 32)?, 32);
    assert_eq!(align_up(33, 32)?, 64);
    Ok(())
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
    let inspection = r.inspect()?;
    assert_eq!(inspection.tensors.len(), 1);
    assert_eq!(inspection.tensors[0].name, "one");
    assert_eq!(inspection.tensors[0].ggml_type, GgmlType::F32);
    assert_eq!(inspection.tensors[0].byte_len, 12);
    Ok(())
}

#[test]
fn whole_file_inspection_returns_known_observed_sha256() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("fixture-with-misleading-suffix.bin");
    let bytes = fixture_bytes();
    std::fs::write(&path, &bytes)?;

    let inspection = inspect_gguf_with_sha256(&path)?;
    let ArtifactDigest::Sha256(digest) = inspection.digest else {
        return GgufSnafu {
            offset: 0u64,
            msg: "whole-file inspection did not return SHA-256".to_string(),
        }
        .fail();
    };
    let expected_file_len = u64::try_from(bytes.len()).map_err(|_| {
        GgufSnafu {
            offset: 0u64,
            msg: "fixture length exceeds u64::MAX".to_string(),
        }
        .build()
    })?;
    assert_eq!(inspection.file_len, expected_file_len);
    assert_eq!(
        digest.to_string(),
        "623d94e17734e71bc68433a1f9121ae9b59f4aabc33fdf74f7b5cc62b61c3980",
        "the digest covers complete fixture bytes, not GGUF metadata only"
    );
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
        return GgufSnafu {
            offset: 0u64,
            msg: "expected MetaValue::Array".to_string(),
        }
        .fail();
    };
    assert_eq!(items.len(), 3);
    assert!(matches!(items[0], MetaValue::U32(10)));
    assert!(matches!(items[1], MetaValue::U32(20)));
    assert!(matches!(items[2], MetaValue::U32(30)));
    Ok(())
}

#[test]
fn array_metadata_round_trips_through_reader_open() -> Result<()> {
    // WHY(forkwright/logismos#37): `array_metadata_type_parses_elements`
    // above exercises `read_meta_value_typed(9)` directly on a bare
    // `Cursor`; nothing before this test combined an array-typed
    // metadata value with the full `Reader::open` path (magic/version/
    // count parsing, the metadata-KV loop's string-key read, and
    // storage into the `metadata()` map). `reads_fixture_bytes` proves
    // that glue for a scalar `U32`; this is the first test to prove it
    // for `MetaValue::Array` — closing an untested *composition*, not
    // a new failure mode. NOTE: the KV loop's insert/lookup
    // (`gguf.rs:220-226`, `reject_duplicate_metadata_key`) is generic
    // over `MetaValue`'s variant, so no fixture can distinguish this
    // test's regression coverage from `reads_fixture_bytes`'s at that
    // seam specifically. Negative-fixture provenance: with the array
    // decode loop's element count deliberately shortened to
    // `0..n.saturating_sub(1)` on a throwaway branch (PR #107, closed
    // without merging, branch deleted after capture), this test failed
    // at the `items.len()` assertion below (`assertion `left == right`
    // failed` / `left: 2` / `right: 3` — CI run 31978345640, job
    // 95241145081) alongside `array_metadata_type_parses_elements`,
    // which fails identically since both bottom out in the same decode
    // loop; the fixture pins that shared loop, not this test's own
    // KV-loop/storage seam. Passes unchanged against the real `0..n`
    // loop.
    let dir = tempdir_for_test();
    let path = dir.join("array-metadata-e2e.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&1u64.to_le_bytes()); // metadata count
    let key = "arr_key";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&9u32.to_le_bytes()); // type = array
    buf.extend_from_slice(&4u32.to_le_bytes()); // inner_type = U32
    buf.extend_from_slice(&3u64.to_le_bytes()); // n = 3 elements
    for v in [1u32, 2, 3] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, buf)?;

    let r = Reader::open(&path)?;
    let Some(MetaValue::Array(items)) = r.metadata().get("arr_key") else {
        return GgufSnafu {
            offset: 0u64,
            msg: "expected metadata()[\"arr_key\"] to be MetaValue::Array".to_string(),
        }
        .fail();
    };
    assert_eq!(items.len(), 3);
    assert!(matches!(items[0], MetaValue::U32(1)));
    assert!(matches!(items[1], MetaValue::U32(2)));
    assert!(matches!(items[2], MetaValue::U32(3)));
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
fn metadata_arrays_have_a_cumulative_element_limit() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&0u32.to_le_bytes()); // inner_type = U8
    buf.extend_from_slice(&1u64.to_le_bytes()); // n = 1
    let mut cur = Cursor::new(&buf);
    cur.total_metadata_array_elements = MAX_TOTAL_METADATA_ARRAY_ELEMENTS;

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
        // u64::MAX`, which `32 bits * u64::MAX` then overflows.
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
        return GgufSnafu {
            offset: 0u64,
            msg: format!("expected Error::Gguf, got {result:?}"),
        }
        .fail();
    };
    assert!(
        msg.contains("byte count overflows"),
        "expected the checked-multiply overflow message, got: {msg}"
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

#[test]
fn inspection_reports_exact_tensor_types_and_serialized_extents() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("hybrid-profile.gguf");
    std::fs::write(&path, hybrid_profile_fixture_bytes()?)?;

    let profile = Reader::open(&path)?.inspect()?;
    assert_eq!(profile.digest, ArtifactDigest::Unverified);
    assert_eq!(profile.model.architecture.as_deref(), Some("qwen3"));
    assert_eq!(profile.model.name.as_deref(), Some("synthetic-hybrid"));
    assert_eq!(profile.model.file_type, Some(12));
    assert_eq!(profile.model.quantization_version, Some(2));
    assert_eq!(profile.tensors.len(), 2);
    assert_eq!(profile.tensors[0].ggml_type, GgmlType::F16);
    assert_eq!(profile.tensors[0].logical_elements, 2);
    assert_eq!(profile.tensors[0].byte_len, 4);
    assert_eq!(profile.tensors[1].ggml_type, GgmlType::Q4K);
    assert_eq!(profile.tensors[1].logical_elements, 256);
    assert_eq!(profile.tensors[1].byte_len, 144);
    assert_eq!(profile.type_census.len(), 2);
    assert_eq!(profile.type_census[0].ggml_type, GgmlType::F16);
    assert_eq!(profile.type_census[1].ggml_type, GgmlType::Q4K);
    Ok(())
}

#[test]
fn inspection_rejects_stale_mmap_after_external_truncation() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("stale-profile.gguf");
    std::fs::write(&path, hybrid_profile_fixture_bytes()?)?;
    let reader = Reader::open(&path)?;
    std::fs::write(&path, b"short")?;

    assert!(matches!(reader.inspect(), Err(Error::MmapStale { .. })));
    Ok(())
}

#[test]
fn whole_file_inspection_rejects_size_changing_concurrent_mutation() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("stale-whole-file-digest.gguf");
    std::fs::write(&path, fixture_bytes())?;
    let reader = Reader::open(&path)?;
    std::fs::write(&path, b"short")?;

    assert!(matches!(
        reader.inspect_with_sha256(),
        Err(Error::MmapStale { .. })
    ));
    Ok(())
}

#[test]
fn reader_rejects_unknown_ggml_type_before_profile_creation() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("unknown-ggml-type.gguf");
    std::fs::write(&path, one_tensor_fixture(999, &[1], 0, 0)?)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn reader_rejects_unsupported_iq_ggml_type_before_profile_creation() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("unsupported-iq-ggml-type.gguf");
    // GGML_TYPE_IQ2_XXS = 16. It must not be mistaken for I8.
    std::fs::write(&path, one_tensor_fixture(16, &[1], 0, 0)?)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn reader_rejects_metadata_count_above_inspection_limit() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("too-many-metadata-entries.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&(MAX_METADATA_COUNT + 1).to_le_bytes());
    std::fs::write(&path, buf)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn reader_rejects_rank_above_ggml_limit() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("too-many-tensor-dimensions.gguf");
    std::fs::write(&path, one_tensor_fixture(0, &[1, 1, 1, 1, 1], 0, 0)?)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn reader_rejects_quantized_tensor_with_partial_block() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("partial-quant-block.gguf");
    std::fs::write(&path, one_tensor_fixture(12, &[1], 0, 0)?)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn reader_rejects_truncated_tensor_payload_during_open() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("truncated-tensor-payload.gguf");
    std::fs::write(&path, one_tensor_fixture(0, &[1], 0, 0)?)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn reader_rejects_descriptor_extent_overflow_during_open() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("extent-overflow.gguf");
    std::fs::write(&path, one_tensor_fixture(0, &[u64::MAX], 0, 0)?)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

#[test]
fn reader_rejects_overlapping_tensor_extents_during_open() -> Result<()> {
    let dir = tempdir_for_test();
    let path = dir.join("overlapping-extents.gguf");
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    append_tensor_descriptor(&mut buf, "first", &[1], 0, 0)?;
    append_tensor_descriptor(&mut buf, "second", &[1], 0, 0)?;
    pad_to_data_region(&mut buf)?;
    buf.extend_from_slice(&0f32.to_le_bytes());
    std::fs::write(&path, buf)?;

    assert!(matches!(Reader::open(&path), Err(Error::Gguf { .. })));
    Ok(())
}

fn hybrid_profile_fixture_bytes() -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    append_string_metadata(&mut buf, "general.architecture", "qwen3")?;
    append_string_metadata(&mut buf, "general.name", "synthetic-hybrid")?;
    append_u32_metadata(&mut buf, "general.file_type", 12)?;
    append_u32_metadata(&mut buf, "general.quantization_version", 2)?;
    append_tensor_descriptor(&mut buf, "blk.0.attn_q.weight", &[2], 1, 0)?;
    append_tensor_descriptor(&mut buf, "blk.0.gdn_a.weight", &[256], 12, 32)?;
    pad_to_data_region(&mut buf)?;
    buf.extend_from_slice(&[0u8; 4]);
    buf.extend_from_slice(&[0u8; 28]);
    buf.extend_from_slice(&[0u8; 144]);
    Ok(buf)
}

fn one_tensor_fixture(
    ggml_type: u32,
    dims: &[u64],
    data_offset: u64,
    payload_len: usize,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    append_tensor_descriptor(&mut buf, "tensor", dims, ggml_type, data_offset)?;
    pad_to_data_region(&mut buf)?;
    buf.extend(std::iter::repeat_n(0u8, payload_len));
    Ok(buf)
}

fn append_string_metadata(buf: &mut Vec<u8>, key: &str, value: &str) -> Result<()> {
    append_string(buf, key)?;
    buf.extend_from_slice(&8u32.to_le_bytes());
    append_string(buf, value)
}

fn append_u32_metadata(buf: &mut Vec<u8>, key: &str, value: u32) -> Result<()> {
    append_string(buf, key)?;
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn append_tensor_descriptor(
    buf: &mut Vec<u8>,
    name: &str,
    dims: &[u64],
    ggml_type: u32,
    data_offset: u64,
) -> Result<()> {
    append_string(buf, name)?;
    let dimension_count = u32::try_from(dims.len()).map_err(|_| {
        GgufSnafu {
            offset: 0u64,
            msg: format!(
                "test tensor dimension count {} exceeds u32::MAX",
                dims.len()
            ),
        }
        .build()
    })?;
    buf.extend_from_slice(&dimension_count.to_le_bytes());
    for dim in dims {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&ggml_type.to_le_bytes());
    buf.extend_from_slice(&data_offset.to_le_bytes());
    Ok(())
}

fn append_string(buf: &mut Vec<u8>, value: &str) -> Result<()> {
    let len = u64::try_from(value.len()).map_err(|_| {
        GgufSnafu {
            offset: 0u64,
            msg: format!("test string length {} exceeds u64::MAX", value.len()),
        }
        .build()
    })?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
    Ok(())
}

fn pad_to_data_region(buf: &mut Vec<u8>) -> Result<()> {
    let header_len = u64::try_from(buf.len()).map_err(|_| {
        GgufSnafu {
            offset: 0u64,
            msg: format!("test GGUF header length {} exceeds u64::MAX", buf.len()),
        }
        .build()
    })?;
    let data_start = align_up(header_len, DEFAULT_ALIGNMENT)?;
    let padding = usize::try_from(data_start - header_len).map_err(|_| {
        GgufSnafu {
            offset: header_len,
            msg: format!(
                "test GGUF padding {} exceeds usize::MAX",
                data_start - header_len
            ),
        }
        .build()
    })?;
    buf.extend(std::iter::repeat_n(0u8, padding));
    Ok(())
}

fn tempdir_for_test() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("logismos-gguf-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&p);
    p
}
