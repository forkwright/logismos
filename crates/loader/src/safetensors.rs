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
use crate::{TensorView, WeightProvider};

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
        // weight directories and doesn't mutate them at inference.
        let mmap = unsafe { Mmap::map(&file)? };

        // Resolve once. We re-parse with `deserialize` to walk tensor
        // entries (read_metadata returns a bare Metadata object that
        // doesn't give us per-tensor byte ranges ergonomically).
        let st = SafeTensors::deserialize(&mmap)?;
        let header_plus_size_prefix = {
            // The header's first 8 bytes are a u64 LE declaring header
            // JSON size; tensor-data starts at 8 + that.
            if mmap.len() < 8 {
                return Err(Error::Safetensors(
                    "file too short for safetensors header".into(),
                ));
            }
            let mut sz = [0u8; 8];
            let prefix = mmap
                .get(..8)
                .ok_or_else(|| Error::Safetensors("missing header length prefix".into()))?;
            sz.copy_from_slice(prefix);
            let header_size_u64 = u64::from_le_bytes(sz);
            let header_size = usize::try_from(header_size_u64).map_err(|_| {
                Error::Safetensors(format!(
                    "safetensors header size {header_size_u64} exceeds usize::MAX"
                ))
            })?;
            header_size.checked_add(8).ok_or_else(|| {
                Error::Safetensors("safetensors header + size prefix overflows usize".into())
            })?
        };

        let mut ordering = Vec::with_capacity(st.len());
        let mut index = HashMap::with_capacity(st.len());
        for (name, tv) in st.iter() {
            let dtype = map_dtype(tv.dtype(), name)?;
            let shape = tv.shape().to_vec();
            let data_ptr = tv.data().as_ptr().addr();
            let mmap_ptr = mmap.as_ptr().addr();
            let start = data_ptr.checked_sub(mmap_ptr).ok_or_else(|| {
                Error::Safetensors(format!("tensor `{name}` data pointer precedes mmap base"))
            })?;
            let end = start.checked_add(tv.data().len()).ok_or_else(|| {
                Error::Safetensors(format!("tensor `{name}` byte range overflows usize"))
            })?;
            ordering.push(name.to_string());
            index.insert(
                name.to_string(),
                Entry {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }

        Ok(Self {
            path: path.to_path_buf(),
            mmap: Arc::new(mmap),
            data_region_start: header_plus_size_prefix,
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
        let entry = self.index.get(name).ok_or_else(|| Error::TensorNotFound {
            name: name.to_string(),
        })?;
        // Key the TensorView name to the stored String in `ordering`
        // so the `&str` is `&self`-bound rather than outlived by the
        // local caller-supplied `name`.
        let stored_name: &str = self.ordering.iter().find(|n| n.as_str() == name).map_or(
            // Should never fire: we just succeeded on
            // `self.index.get(name)`.
            self.ordering.first().map_or("", String::as_str),
            String::as_str,
        );
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
mod tests {
    use std::collections::HashMap as StdMap;

    use safetensors::serialize_to_file;
    use safetensors::tensor::TensorView as UpstreamView;

    use super::*;

    fn write_tiny_fixture(path: &Path) -> Result<()> {
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
        let tv_a = r.get("a")?;
        assert_eq!(tv_a.dtype, taxis::DType::F32);
        assert_eq!(tv_a.shape, vec![2, 2]);
        assert_eq!(tv_a.bytes.len(), 16);
        let tensor = tv_a.to_tensor_cpu()?;
        assert_eq!(tensor.dims(), &[2, 2]);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }
}
