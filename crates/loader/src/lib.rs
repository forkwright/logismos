//! # loader
//!
//! Weight loaders for the logismos platform.
//!
//! Two input formats are supported:
//!
//! - **safetensors** — via the upstream [`safetensors`] crate wrapped
//!   in [`crate::safetensors::Reader`].
//! - **GGUF** — via our own v3 reader in [`gguf::Reader`] (metadata +
//!   a tensor index; fp16/bf16/f32 tensor bytes only — K-quant blocks
//!   are Phase-6 work per the PLAN).
//!
//! Both formats expose a common [`WeightProvider`] trait and resolve
//! through an [`Archive`] enum so consumers can write format-agnostic
//! code. A small [`mapping::NameMap`] utility translates model-family
//! HF names onto logismos-internal keys; tables live in each model's
//! module under `decoders` / `encoders` / `embed`.
//!
//! Phase 2 scope:
//! - Read metadata + tensor bytes.
//! - Copy a tensor into a CPU `taxis::Tensor`.
//! - HIP-side upload is a separate, one-line caller step via
//!   `taxis::Tensor::from_host_f32` / `_f16` / `_bf16`.
//!
//! Non-goals for Phase 2: K-quant dequantisation, lazy mmap paging for
//! tensors larger than host RAM, and network fetches.
//!
//! Rationale for using upstream `safetensors`: the format is a public
//! HF standard; its parser is the API boundary, not the value. See
//! Phase-0 dossier and Phase-2 PLAN §`loader`.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::elidable_lifetime_names
)]

pub mod error;
pub mod gguf;
pub mod mapping;
pub mod provider;
pub mod safetensors;

pub use crate::error::{Error, Result};
pub use crate::mapping::NameMap;
pub use crate::provider::WeightProvider;

/// Read-only view of a single tensor inside an archive.
///
/// Views borrow from their archive and do not own memory. Callers copy
/// into a `taxis::Tensor` with [`TensorView::to_tensor_cpu`].
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    /// Logical name (verbatim from the archive; no normalisation).
    pub name: &'a str,
    /// Runtime dtype.
    pub dtype: taxis::DType,
    /// Shape in logical element counts.
    pub shape: Vec<usize>,
    /// Raw contiguous bytes in the archive's native layout (row-major).
    pub bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Copy this view into a newly allocated CPU `taxis::Tensor`.
    ///
    /// Only the dtypes that the Phase-1 compute path recognises land
    /// here (`F32`, `F16`, `BF16`, `I32`, `I8`, `U8`). Anything else
    /// returns [`Error::UnsupportedDType`]. The safetensors format
    /// itself does carry more dtypes, but we refuse to silently widen
    /// — the caller decides which transforms are lossless.
    pub fn to_tensor_cpu(&self) -> Result<taxis::Tensor> {
        use taxis::{CpuStorage, Shape, Tensor};

        let shape = Shape::new(&self.shape);
        let elem_count = shape.elem_count();
        let expected_bytes = self.dtype.byte_count(elem_count);
        if self.bytes.len() != expected_bytes {
            return Err(Error::ShapeMismatch {
                name: self.name.to_string(),
                dtype: self.dtype,
                elem_count,
                expected_bytes,
                actual_bytes: self.bytes.len(),
            });
        }

        let storage = match self.dtype {
            taxis::DType::F32 => CpuStorage::F32(read_f32_le(self.bytes)),
            taxis::DType::F16 => CpuStorage::F16(read_f16_le(self.bytes)),
            taxis::DType::BF16 => CpuStorage::BF16(read_bf16_le(self.bytes)),
            taxis::DType::I32 => CpuStorage::I32(read_i32_le(self.bytes)),
            taxis::DType::I8 => CpuStorage::I8(read_i8_le(self.bytes)),
            taxis::DType::U8 => CpuStorage::U8(self.bytes.to_vec()),
            other => {
                return Err(Error::UnsupportedDType {
                    name: self.name.to_string(),
                    dtype: other,
                });
            }
        };

        Ok(Tensor::from_cpu(storage, shape))
    }
}

fn read_i8_le(bytes: &[u8]) -> Vec<i8> {
    bytes
        .chunks_exact(1)
        .map(|c| {
            let mut b = [0u8; 1];
            b.copy_from_slice(c);
            i8::from_ne_bytes(b)
        })
        .collect()
}

fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b.copy_from_slice(c);
            f32::from_le_bytes(b)
        })
        .collect()
}

fn read_i32_le(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b.copy_from_slice(c);
            i32::from_le_bytes(b)
        })
        .collect()
}

fn read_f16_le(bytes: &[u8]) -> Vec<half::f16> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let mut b = [0u8; 2];
            b.copy_from_slice(c);
            half::f16::from_le_bytes(b)
        })
        .collect()
}

fn read_bf16_le(bytes: &[u8]) -> Vec<half::bf16> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let mut b = [0u8; 2];
            b.copy_from_slice(c);
            half::bf16::from_le_bytes(b)
        })
        .collect()
}

/// Sum type over the supported archive formats.
#[non_exhaustive]
pub enum Archive {
    /// A safetensors archive (upstream reader).
    Safetensors(crate::safetensors::Reader),
    /// A GGUF v3 archive (our own parser).
    Gguf(crate::gguf::Reader),
}

impl Archive {
    /// Open an archive by format auto-detection: `.safetensors` → safetensors,
    /// `.gguf` → GGUF (case-insensitively), everything else returns
    /// [`Error::UnknownFormat`].
    ///
    /// # Errors
    ///
    /// [`Error::UnknownFormat`] when the extension matches neither
    /// format; otherwise propagates whatever the matched format's
    /// `Reader::open` returns.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        // WHY(forkwright/logismos#60): matching the raw extension
        // rejected a valid `.GGUF` or `.SafeTensors` file as unknown —
        // case doesn't carry format information on any filesystem this
        // loader targets.
        let ext = path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("safetensors") => Ok(Self::Safetensors(crate::safetensors::Reader::open(path)?)),
            Some("gguf") => Ok(Self::Gguf(crate::gguf::Reader::open(path)?)),
            _ => Err(Error::UnknownFormat {
                path: path.to_path_buf(),
            }),
        }
    }
}

/// Re-stat `path` and error if its length no longer matches
/// `expected_len` (the mmap's length at open time).
///
/// Best-effort: this check and the slice read that follows it are not
/// atomic, so it closes the common non-adversarial case — a weights
/// file re-saved or truncated by something else while a `Reader` still
/// has it mapped — rather than a fully adversarial race. See the
/// SAFETY comments on `gguf::Reader::open` / `safetensors::Reader::open`
/// for what this does and doesn't guarantee.
///
/// # Errors
///
/// [`Error::Io`] if the re-stat fails; [`Error::MmapStale`] if the
/// file's current length disagrees with `expected_len`.
pub(crate) fn check_mmap_not_truncated(path: &std::path::Path, expected_len: usize) -> Result<()> {
    let actual_len = std::fs::metadata(path)?.len();
    let expected_len = u64::try_from(expected_len)
        .map_err(|_| Error::Msg(format!("mmap length {expected_len} exceeds u64::MAX")))?;
    if actual_len != expected_len {
        return Err(Error::MmapStale {
            path: path.to_path_buf(),
            expected_len,
            actual_len,
        });
    }
    Ok(())
}

impl WeightProvider for Archive {
    fn get(&self, name: &str) -> Result<TensorView<'_>> {
        match self {
            Self::Safetensors(r) => r.get(name),
            Self::Gguf(r) => r.get(name),
        }
    }
    fn names(&self) -> Vec<String> {
        match self {
            Self::Safetensors(r) => r.names(),
            Self::Gguf(r) => r.names(),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Safetensors(r) => r.len(),
            Self::Gguf(r) => r.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_view_rejects_mismatched_byte_count() {
        let view = TensorView {
            name: "bad",
            dtype: taxis::DType::F32,
            shape: vec![2],
            bytes: &[0, 0, 0, 0],
        };
        assert!(matches!(
            view.to_tensor_cpu(),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn to_tensor_cpu_converts_every_supported_dtype() -> Result<()> {
        // WHY(forkwright/logismos#56): only F32 had coverage; the other
        // five branches of the match in `to_tensor_cpu` had none.
        let cases: &[(taxis::DType, usize)] = &[
            (taxis::DType::F16, 2),
            (taxis::DType::BF16, 2),
            (taxis::DType::I8, 1),
            (taxis::DType::I32, 4),
            (taxis::DType::U8, 1),
        ];
        for &(dtype, bytes_per_elem) in cases {
            let bytes = vec![0u8; bytes_per_elem * 3];
            let view = TensorView {
                name: "t",
                dtype,
                shape: vec![3],
                bytes: &bytes,
            };
            let tensor = view.to_tensor_cpu()?;
            assert_eq!(tensor.dtype(), dtype);
            assert_eq!(tensor.dims(), &[3]);
        }
        Ok(())
    }

    #[test]
    fn to_tensor_cpu_rejects_unsupported_dtype() {
        // WHY(forkwright/logismos#56): the `UnsupportedDType` fallback
        // arm had no test constructing it.
        let bytes = [0u8; 3];
        let view = TensorView {
            name: "unsupported",
            dtype: taxis::DType::F8E4M3,
            shape: vec![3],
            bytes: &bytes,
        };
        assert!(matches!(
            view.to_tensor_cpu(),
            Err(Error::UnsupportedDType { .. })
        ));
    }

    #[test]
    fn archive_open_dispatches_by_extension_case_insensitively() -> Result<()> {
        // WHY(forkwright/logismos#56): `Archive::open`'s dispatch and
        // its `UnknownFormat` error had no test at all. WHY(#60): the
        // uppercase extensions here are the case-insensitivity fix's
        // own negative-turned-positive fixture — before it, `.GGUF` /
        // `.SafeTensors` fell into `UnknownFormat`.
        let dir =
            std::env::temp_dir().join(format!("logismos-archive-open-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let gguf_path = dir.join("model.GGUF");
        std::fs::write(&gguf_path, crate::gguf::tests::fixture_bytes())?;
        assert!(matches!(Archive::open(&gguf_path)?, Archive::Gguf(_)));

        let st_path = dir.join("model.SafeTensors");
        crate::safetensors::tests::write_tiny_fixture(&st_path)?;
        assert!(matches!(Archive::open(&st_path)?, Archive::Safetensors(_)));

        Ok(())
    }

    #[test]
    fn archive_open_rejects_unknown_extension() {
        let path = std::path::Path::new("/nonexistent/model.bin");
        assert!(matches!(
            Archive::open(path),
            Err(Error::UnknownFormat { .. })
        ));
    }

    #[test]
    fn check_mmap_not_truncated_detects_size_change() -> Result<()> {
        // Negative fixture for the mmap-staleness guard used by both
        // readers' `get()`: a length match passes, a length mismatch
        // (the file shrinking after `expected_len` was captured) must
        // return `Error::MmapStale`.
        let path = std::env::temp_dir().join(format!(
            "logismos-truncation-test-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, b"0123456789")?;
        check_mmap_not_truncated(&path, 10)?;

        std::fs::write(&path, b"012")?;
        let result = check_mmap_not_truncated(&path, 10);
        assert!(matches!(result, Err(Error::MmapStale { .. })));

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}
