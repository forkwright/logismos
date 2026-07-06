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
    /// `.gguf` → GGUF, everything else returns [`Error::UnknownFormat`].
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let ext = path.extension().and_then(std::ffi::OsStr::to_str);
        match ext {
            Some("safetensors") => Ok(Self::Safetensors(crate::safetensors::Reader::open(path)?)),
            Some("gguf") => Ok(Self::Gguf(crate::gguf::Reader::open(path)?)),
            _ => Err(Error::UnknownFormat {
                path: path.to_path_buf(),
            }),
        }
    }
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
}
