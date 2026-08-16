//! Safetensors reader, wrapping the upstream `safetensors` crate.
//!
//! The upstream crate's `SafeTensors<'data>` borrows from the archive's
//! raw byte buffer. Rather than hold a self-referential struct (parser
//! + mmap) we do the following:
//!
//! 1. Open the file, mmap it.
//! 2. Call `SafeTensors::read_metadata` to resolve the header once.
//! 3. Build an owned `HashMap<name, (dtype, shape, byte_range_in_mmap)>`
//!    index.
//! 4. `get(name)` then slices the mmap directly, returning a
//!    `TensorView<'_>` tied to `&self`.
//!
//! This avoids repeated header parsing and lets the caller hold a
//! view whose lifetime ties cleanly to the `Reader`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype as UpstreamDtype;

use crate::error::{Error, Result};
use crate::{TensorView, WeightProvider, check_mmap_not_truncated};

/// The header's first 8 bytes are a little-endian `u64` declaring the
/// JSON header's own byte length; `SafeTensors::read_metadata` reports
/// that length alone, so the tensor-data region starts this many bytes
/// further in.
const HEADER_LEN_PREFIX: usize = 8;

/// Owning safetensors archive.
pub struct Reader {
    path: PathBuf,
    mmap: Arc<Mmap>,
    /// Byte range in the file where the tensor-data region begins.
    data_region_start: usize,
    /// In-order tensor names (archive ordering).
    ordering: Vec<String>,
    /// Per-tensor metadata, keyed by name.
    index: HashMap<String, Entry>,
}

#[derive(Debug, Clone)]
struct Entry {
    dtype: taxis::DType,
    shape: Vec<usize>,
    /// Byte-range within the whole mmap (absolute).
    start: usize,
    end: usize,
}

impl Reader {
    /// Open a safetensors file from disk.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] on fs / mmap failure, [`Error::Safetensors`] if
    /// the upstream parser rejects the header.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: mmap is sound for a read-only mapping as long as the
        // underlying file is not mutated under us. logismos owns its
        // weight directories and doesn't mutate them at inference; `get`
        // additionally re-stats the file before every access as a
        // best-effort guard against a mapping outliving the file it was
        // taken from (see `check_mmap_not_truncated`) — that narrows,
        // but cannot close, the race between an external writer and a
        // concurrent read of the mapped bytes.
        let mmap = unsafe { Mmap::map(&file)? };

        // WHY(forkwright/logismos#56): the previous implementation
        // called `SafeTensors::deserialize` and then recovered each
        // tensor's byte range by subtracting `TensorView::data().as_ptr()`
        // from the mmap's base pointer. That's brittle against upstream
        // changes: nothing in the API contracts that `data()` returns a
        // pointer into the original buffer rather than a copy, and
        // pointer-address subtraction is meaningless if it ever doesn't.
        // `SafeTensors::read_metadata` performs the identical validation
        // `deserialize` does (offsets monotonic/non-overlapping, each
        // tensor's byte length matches its declared dtype × shape, and
        // the total matches the buffer length) but hands back the parsed
        // `Metadata`, whose `TensorInfo::data_offsets` is the offset
        // pair directly — no pointer arithmetic required.
        let (header_size, metadata) = SafeTensors::read_metadata(&mmap)?;
        let data_region_start = header_size.checked_add(HEADER_LEN_PREFIX).ok_or_else(|| {
            Error::Safetensors("safetensors header + size prefix overflows usize".into())
        })?;

        // `offset_keys()` orders names by on-disk tensor offset, which
        // is what `ordering`'s doc comment ("archive ordering") means.
        let ordering = metadata.offset_keys();
        let mut index = HashMap::with_capacity(ordering.len());
        for name in &ordering {
            let info = metadata.info(name).ok_or_else(|| {
                Error::Safetensors(format!(
                    "tensor `{name}` in offset_keys but missing from metadata"
                ))
            })?;
            let dtype = map_dtype(info.dtype, name)?;
            let (rel_start, rel_end) = info.data_offsets;
            let start = data_region_start.checked_add(rel_start).ok_or_else(|| {
                Error::Safetensors(format!("tensor `{name}` start offset overflows usize"))
            })?;
            let end = data_region_start.checked_add(rel_end).ok_or_else(|| {
                Error::Safetensors(format!("tensor `{name}` end offset overflows usize"))
            })?;
            index.insert(
                name.clone(),
                Entry {
                    dtype,
                    shape: info.shape.clone(),
                    start,
                    end,
                },
            );
        }

        Ok(Self {
            path: path.to_path_buf(),
            mmap: Arc::new(mmap),
            data_region_start,
            ordering,
            index,
        })
    }

    /// Path the reader was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Byte offset into the file where tensor-data starts.
    pub fn data_region_start(&self) -> usize {
        self.data_region_start
    }
}

impl WeightProvider for Reader {
    fn get(&self, name: &str) -> Result<TensorView<'_>> {
        check_mmap_not_truncated(&self.path, self.mmap.len())?;
        // `get_key_value` ties the returned name to the `String` owned
        // by `self.index` (== `&self`-bound) in one lookup, rather than
        // a second linear scan through `ordering` to recover it.
        let (stored_name, entry) =
            self.index
                .get_key_value(name)
                .ok_or_else(|| Error::TensorNotFound {
                    name: name.to_string(),
                })?;
        let bytes = self.mmap.get(entry.start..entry.end).ok_or_else(|| {
            Error::Safetensors(format!(
                "tensor `{name}` byte range [{start}..{end}] out of mmap bounds",
                start = entry.start,
                end = entry.end,
            ))
        })?;
        Ok(TensorView {
            name: stored_name,
            dtype: entry.dtype,
            shape: entry.shape.clone(),
            bytes,
        })
    }

    fn names(&self) -> Vec<String> {
        self.ordering.clone()
    }

    fn len(&self) -> usize {
        self.ordering.len()
    }
}

fn map_dtype(d: UpstreamDtype, name: &str) -> Result<taxis::DType> {
    Ok(match d {
        UpstreamDtype::F32 => taxis::DType::F32,
        UpstreamDtype::F16 => taxis::DType::F16,
        UpstreamDtype::BF16 => taxis::DType::BF16,
        UpstreamDtype::I32 => taxis::DType::I32,
        UpstreamDtype::I8 => taxis::DType::I8,
        UpstreamDtype::U8 => taxis::DType::U8,
        UpstreamDtype::F8_E4M3 => taxis::DType::F8E4M3,
        UpstreamDtype::F8_E5M2 => taxis::DType::F8E5M2,
        other => {
            return Err(Error::Msg(format!(
                "tensor `{name}` uses unmapped safetensors dtype {other:?}"
            )));
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap as StdMap;

    use safetensors::serialize_to_file;
    use safetensors::tensor::TensorView as UpstreamView;

    use super::*;

    /// `pub(crate)` so `lib.rs`'s `Archive::open` dispatch test can
    /// reuse it rather than duplicating a fixture builder.
    pub(crate) fn write_tiny_fixture(path: &Path) -> Result<()> {
        // Two tiny F32 tensors, written via the upstream serializer so
        // the test exercises the real on-disk layout.
        let a: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let b: Vec<u8> = [10.0_f32, 20.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let tv_a = UpstreamView::new(UpstreamDtype::F32, vec![2, 2], &a)
            .map_err(|e| Error::Msg(format!("tv_a: {e}")))?;
        let tv_b = UpstreamView::new(UpstreamDtype::F32, vec![2], &b)
            .map_err(|e| Error::Msg(format!("tv_b: {e}")))?;
        let mut tensors: StdMap<String, UpstreamView<'_>> = StdMap::new();
        tensors.insert("a".into(), tv_a);
        tensors.insert("b".into(), tv_b);
        serialize_to_file(&tensors, None, path)
            .map_err(|e| Error::Msg(format!("serialize_to_file: {e}")))?;
        Ok(())
    }

    #[test]
    fn reads_fixture() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-st-test-{}.safetensors",
            std::process::id()
        ));
        write_tiny_fixture(&tmp)?;

        let r = Reader::open(&tmp)?;
        let names = r.names();
        assert!(names.iter().any(|n| n == "a"));
        assert!(names.iter().any(|n| n == "b"));

        // WHY(forkwright/logismos#56): the offset-computation rewrite
        // (pointer-address subtraction -> `TensorInfo::data_offsets`)
        // is only pinned if the actual decoded bytes are checked — a
        // length-only assertion passes on a slice shifted by any
        // constant, since `rel_end - rel_start` is invariant to a
        // shift in `data_region_start`. Both fixture tensors are
        // checked so an error confined to a non-first tensor (e.g.
        // cumulative-offset drift) is also caught.
        let expected_a: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let expected_b: Vec<u8> = [10.0_f32, 20.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tv_a = r.get("a")?;
        assert_eq!(tv_a.dtype, taxis::DType::F32);
        assert_eq!(tv_a.shape, vec![2, 2]);
        assert_eq!(tv_a.bytes, expected_a.as_slice());

        let tv_b = r.get("b")?;
        assert_eq!(tv_b.dtype, taxis::DType::F32);
        assert_eq!(tv_b.shape, vec![2]);
        assert_eq!(tv_b.bytes, expected_b.as_slice());

        let tensor = tv_a.to_tensor_cpu()?;
        assert_eq!(tensor.dims(), &[2, 2]);
        let Some(taxis::CpuStorage::F32(v)) = tensor.cpu_storage() else {
            return Err(Error::Msg("expected F32 CPU storage".into()));
        };
        assert_eq!(v.as_slice(), &[1.0_f32, 2.0, 3.0, 4.0][..]);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn get_rejects_after_external_truncation() -> Result<()> {
        // WHY(forkwright/logismos#60): the mmap SAFETY comment states
        // the backing file must not be mutated while the mapping is
        // open; nothing enforced that. `get` now re-stats the file and
        // refuses a mapping whose backing file has changed size since
        // it was opened.
        let tmp = std::env::temp_dir().join(format!(
            "logismos-st-truncation-test-{}.safetensors",
            std::process::id()
        ));
        write_tiny_fixture(&tmp)?;
        let r = Reader::open(&tmp)?;
        std::fs::write(&tmp, b"short")?;

        let result = r.get("a");
        assert!(matches!(result, Err(Error::MmapStale { .. })));

        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }
}
