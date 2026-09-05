//! GGUF v3 reader — our own implementation.
//!
//! Parses:
//! - Magic (`"GGUF"`) + version (3).
//! - `u64` tensor count + `u64` metadata-KV count.
//! - Metadata KV pairs (all 13 value types per the spec).
//! - Tensor descriptors (name + dims + ggml-type + data-offset).
//!
//! The tensor-data payload is mmap-backed and lookups return a
//! borrowed `&[u8]` slice anchored on the mmap.
//!
//! ## Scope
//!
//! Phase 2 supports ggml types that are either:
//! - `F32` (id 0),
//! - `F16` (id 1),
//! - `BF16` (id 30),
//! - `I8` / `I16` / `I32` / `I64` metadata tensors.
//!
//! K-quant blocks (`Q4_K`, `Q6_K`, etc.) deliberately error out with a
//! clear message — they land in Phase 6 when the quant kernels + block
//! decoders ship. This keeps the surface honest: we don't pretend to
//! decode bytes whose layout we can't yet apply.
//!
//! Spec reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//! (GGUF v3; magic 0x46554747, little-endian throughout).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{GgufSnafu, Result, TensorNotFoundSnafu};
// WHY imported without a code reference: the `# Errors` sections below link to
// `Error` variants by intra-doc path, which rustdoc resolves only against items
// in scope. Split from the group above so the expectation covers this import
// alone -- a later genuinely-unused import in the group still fails the gate.
#[expect(
    unused_imports,
    reason = "resolves intra-doc links in this module's `# Errors` sections"
)]
use crate::error::Error;
use crate::{TensorView, WeightProvider, check_mmap_not_truncated};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_V3: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_METADATA_COUNT: u64 = 100_000;
const MAX_METADATA_ARRAY_ELEMENTS: u64 = 262_144;
const MAX_STRING_BYTES: u64 = 1 << 20;
const MAX_TOTAL_STRING_BYTES: u64 = 64 << 20;
const MAX_TENSOR_DIMS: u32 = 4;
const MIN_METADATA_ENTRY_BYTES: u64 = 13;
const MIN_TENSOR_DESCRIPTOR_BYTES: u64 = 24;

/// ggml / GGUF dtype tag. Numeric ids match the spec verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
#[expect(
    missing_docs,
    reason = "variant names map 1:1 to ggml spec dtype tags (see file header)"
)]
#[non_exhaustive]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(id: u32, offset: u64) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::I8,
            17 => Self::I16,
            18 => Self::I32,
            19 => Self::I64,
            20 => Self::F64,
            30 => Self::BF16,
            other => {
                return GgufSnafu {
                    offset,
                    msg: format!("unknown ggml type id {other}"),
                }
                .fail();
            }
        })
    }

    /// Size of one logical element in bits for types that have a native
    /// per-element size. Block-quant types return `None` because their
    /// on-disk layout is block-struct-per-group-of-elements and a
    /// per-element bit-width is ill-defined.
    fn size_in_bits(self) -> Option<usize> {
        Some(match self {
            Self::F32 | Self::I32 => 32,
            Self::F16 | Self::BF16 | Self::I16 => 16,
            Self::I8 => 8,
            Self::I64 | Self::F64 => 64,
            _ => return None,
        })
    }

    fn block_layout(self) -> Option<(u64, u64)> {
        Some(match self {
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            _ => return None,
        })
    }

    fn to_taxis_dtype(self) -> Result<taxis::DType> {
        Ok(match self {
            Self::F32 => taxis::DType::F32,
            Self::F16 => taxis::DType::F16,
            Self::BF16 => taxis::DType::BF16,
            Self::I32 => taxis::DType::I32,
            Self::I8 => taxis::DType::I8,
            other => {
                return GgufSnafu {
                    offset: 0u64,
                    msg: format!(
                        "ggml type {other:?} not decodable in Phase 2 \
                         (K-quant / block dtypes land with quant kernels in Phase 6)"
                    ),
                }
                .fail();
            }
        })
    }
}

/// GGUF metadata value. All 13 spec value types.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MetaValue {
    /// 8-bit unsigned.
    U8(u8),
    /// 8-bit signed.
    I8(i8),
    /// 16-bit unsigned.
    U16(u16),
    /// 16-bit signed.
    I16(i16),
    /// 32-bit unsigned.
    U32(u32),
    /// 32-bit signed.
    I32(i32),
    /// 64-bit unsigned.
    U64(u64),
    /// 64-bit signed.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean (one byte, nonzero = true).
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Typed array.
    Array(Vec<MetaValue>),
}

/// Per-tensor header entry.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Tensor name.
    pub name: String,
    /// Dimensions in logical elements.
    pub dims: Vec<u64>,
    /// ggml type.
    pub ggml_type: GgmlType,
    /// Offset from the start of the tensor-data region.
    pub data_offset: u64,
}

/// The inspection result has no cryptographic artifact identity.
///
/// SHA-256 verification is intentionally external to this bounded metadata
/// reader; a path or mmap must not be presented as immutable identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactDigest {
    /// The caller has not supplied separately verified digest evidence.
    Unverified,
}

/// Stable model-level metadata selected from the GGUF key/value table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelMetadata {
    /// Value of `general.architecture`, when stored as a string.
    pub architecture: Option<String>,
    /// Value of `general.name`, when stored as a string.
    pub name: Option<String>,
    /// Value of `general.file_type`, when stored as an unsigned 32-bit integer.
    pub file_type: Option<u32>,
    /// Value of `general.quantization_version`, when stored as an unsigned 32-bit integer.
    pub quantization_version: Option<u32>,
}

/// One tensor's validated on-disk GGUF extent.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InspectedTensor {
    /// Tensor name verbatim from the archive.
    pub name: String,
    /// Dimensions in logical elements.
    pub dims: Vec<u64>,
    /// Exact GGML storage type from the tensor descriptor.
    pub ggml_type: GgmlType,
    /// Checked product of `dims`.
    pub logical_elements: u64,
    /// Absolute byte offset in the GGUF file.
    pub file_offset: u64,
    /// Serialized byte length, not a runtime-memory estimate.
    pub byte_len: u64,
}

/// Aggregate for one exact GGML storage type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GgmlTypeCensus {
    /// Exact GGML storage type.
    pub ggml_type: GgmlType,
    /// Number of tensors using this type.
    pub tensor_count: u64,
    /// Sum of checked logical element counts for this type.
    pub logical_elements: u64,
    /// Sum of validated serialized byte lengths for this type.
    pub byte_len: u64,
}

/// CPU-only, parse-derived GGUF artifact profile.
///
/// The profile identifies exact descriptor types and on-disk extents. It does
/// not decode tensor payloads, reserve device memory, classify execution
/// support, or establish a cryptographic artifact identity.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Inspection {
    /// GGUF v3 file length observed while the archive is mapped.
    pub file_len: u64,
    /// Alignment used to locate the tensor-data region.
    pub alignment: u64,
    /// The report's cryptographic-identity state.
    pub digest: ArtifactDigest,
    /// Bounded selection of model metadata.
    pub model: ModelMetadata,
    /// Per-tensor exact type, shape, and validated serialized extent.
    pub tensors: Vec<InspectedTensor>,
    /// Exact-type aggregate derived from [`Inspection::tensors`].
    pub type_census: Vec<GgmlTypeCensus>,
}

/// Owning GGUF archive.
pub struct Reader {
    path: PathBuf,
    mmap: Arc<Mmap>,
    metadata: HashMap<String, MetaValue>,
    tensors: Vec<TensorDescriptor>,
    tensor_by_name: HashMap<String, usize>,
    data_region_start: u64,
    alignment: u64,
}

impl Reader {
    /// Open a GGUF file from disk.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] on fs / mmap failure; [`Error::Gguf`] on a
    /// malformed header, an unsupported version, a parser resource limit,
    /// a duplicate metadata key, a duplicate tensor name, an invalid tensor
    /// layout, or an out-of-file/overlapping tensor extent.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: see `safetensors::Reader::open`.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        let mut cur = Cursor::new(&mmap);
        cur.check_magic()?;
        let version = cur.read_u32()?;
        if version != GGUF_V3 {
            return GgufSnafu {
                offset: cur.pos,
                msg: format!("unsupported GGUF version {version} (target: v{GGUF_V3})"),
            }
            .fail();
        }
        let tensor_count = cur.read_u64()?;
        let metadata_count = cur.read_u64()?;
        validate_header_counts(&cur, tensor_count, metadata_count)?;

        let mut metadata: HashMap<String, MetaValue> = HashMap::new();
        for _ in 0..metadata_count {
            let key = cur.read_string()?;
            let val = cur.read_meta_value()?;
            reject_duplicate_metadata_key(&metadata, &key, cur.pos)?;
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(MetaValue::U32(v)) => u64::from(*v),
            _ => DEFAULT_ALIGNMENT,
        };
        validate_alignment(alignment, cur.pos)?;

        // WHY(forkwright/logismos#34): `tensor_count` is an untrusted u64
        // straight off the wire. `usize::try_from` only rejects values
        // that overflow `usize` (never, on a 64-bit target), so a file
        // claiming e.g. 10^18 tensors would otherwise drive
        // `Vec::with_capacity` to request an allocation the global
        // allocator cannot satisfy — which aborts the process rather than
        // returning an `Err`. No pre-allocation: `Vec::push` amortises
        // its own growth, and `cur.read_string()`/`cur.read_u32()` below
        // already bounds-check against the mmap, so a claimed count that
        // outruns the real file fails fast with `Error::Gguf` instead.
        let mut tensors = Vec::new();
        let mut tensor_by_name = HashMap::new();
        for i in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()?;
            if !(1..=MAX_TENSOR_DIMS).contains(&n_dims) {
                return GgufSnafu {
                    offset: cur.pos,
                    msg: format!(
                        "tensor `{name}` has {n_dims} dimensions; expected 1..={MAX_TENSOR_DIMS}"
                    ),
                }
                .fail();
            }
            // WHY(forkwright/logismos#34): same rationale as tensor_count
            // above — no pre-allocation from an untrusted count.
            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            reject_zero_length_dimension(&name, &dims, cur.pos)?;
            let ggml_type_id = cur.read_u32()?;
            let ggml_type = GgmlType::from_u32(ggml_type_id, cur.pos)?;
            let data_offset = cur.read_u64()?;
            let idx_usize = usize::try_from(i).map_err(|_| {
                GgufSnafu {
                    offset: cur.pos,
                    msg: format!("tensor index {i} exceeds usize::MAX"),
                }
                .build()
            })?;
            reject_duplicate_tensor_name(&tensor_by_name, &name, cur.pos)?;
            tensor_by_name.insert(name.clone(), idx_usize);
            tensors.push(TensorDescriptor {
                name,
                dims,
                ggml_type,
                data_offset,
            });
        }

        // Data region starts at the next alignment boundary after the
        // header cursor.
        let after_header = cur.pos;
        let data_region_start = align_up(after_header, alignment)?;

        let reader = Self {
            path: path.to_path_buf(),
            mmap,
            metadata,
            tensors,
            tensor_by_name,
            data_region_start,
            alignment,
        };
        reader.validate_tensor_extents()?;
        Ok(reader)
    }

    /// Path opened.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Alignment used for the tensor-data region.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Offset in the file where tensor-data starts.
    pub fn data_region_start(&self) -> u64 {
        self.data_region_start
    }

    /// Read-only view of the metadata map.
    pub fn metadata(&self) -> &HashMap<String, MetaValue> {
        &self.metadata
    }

    /// Full list of tensor descriptors in file order.
    pub fn tensor_descriptors(&self) -> &[TensorDescriptor] {
        &self.tensors
    }

    /// Inspect the mapped artifact without decoding tensor payloads or using HIP.
    ///
    /// # Errors
    ///
    /// [`Error::MmapStale`] if the mapped file's size changed after open, or
    /// [`Error::Gguf`] if a checked aggregate overflows.
    pub fn inspect(&self) -> Result<Inspection> {
        check_mmap_not_truncated(&self.path, self.mmap.len())?;
        let mut census = BTreeMap::new();
        let mut tensors = Vec::new();
        for desc in &self.tensors {
            let extent = self.extent_for(desc)?;
            let entry = census.entry(desc.ggml_type).or_insert((0u64, 0u64, 0u64));
            entry.0 = entry.0.checked_add(1).ok_or_else(|| {
                GgufSnafu {
                    offset: extent.start,
                    msg: "GGUF type census tensor count overflows u64".to_string(),
                }
                .build()
            })?;
            entry.1 = entry.1.checked_add(extent.elements).ok_or_else(|| {
                GgufSnafu {
                    offset: extent.start,
                    msg: format!(
                        "GGUF type census element count overflows u64 for {:?}",
                        desc.ggml_type
                    ),
                }
                .build()
            })?;
            entry.2 = entry.2.checked_add(extent.byte_len).ok_or_else(|| {
                GgufSnafu {
                    offset: extent.start,
                    msg: format!(
                        "GGUF type census byte count overflows u64 for {:?}",
                        desc.ggml_type
                    ),
                }
                .build()
            })?;
            tensors.push(InspectedTensor {
                name: desc.name.clone(),
                dims: desc.dims.clone(),
                ggml_type: desc.ggml_type,
                logical_elements: extent.elements,
                file_offset: extent.start,
                byte_len: extent.byte_len,
            });
        }
        let type_census = census
            .into_iter()
            .map(
                |(ggml_type, (tensor_count, logical_elements, byte_len))| GgmlTypeCensus {
                    ggml_type,
                    tensor_count,
                    logical_elements,
                    byte_len,
                },
            )
            .collect();

        Ok(Inspection {
            file_len: u64::try_from(self.mmap.len()).map_err(|_| {
                GgufSnafu {
                    offset: 0,
                    msg: format!("GGUF file length {} exceeds u64::MAX", self.mmap.len()),
                }
                .build()
            })?,
            alignment: self.alignment,
            digest: ArtifactDigest::Unverified,
            model: ModelMetadata {
                architecture: metadata_string(&self.metadata, "general.architecture"),
                name: metadata_string(&self.metadata, "general.name"),
                file_type: metadata_u32(&self.metadata, "general.file_type"),
                quantization_version: metadata_u32(&self.metadata, "general.quantization_version"),
            },
            tensors,
            type_census,
        })
    }
}

impl Reader {
    /// Look up a tensor descriptor by name.
    fn descriptor_by_name(&self, name: &str) -> Result<&TensorDescriptor> {
        let idx = self.tensor_by_name.get(name).copied().ok_or_else(|| {
            TensorNotFoundSnafu {
                name: name.to_string(),
            }
            .build()
        })?;
        self.tensors.get(idx).ok_or_else(|| {
            GgufSnafu {
                offset: 0u64,
                msg: format!(
                    "internal: tensor index {idx} out of range (len={})",
                    self.tensors.len()
                ),
            }
            .build()
        })
    }

    fn extent_for(&self, desc: &TensorDescriptor) -> Result<TensorExtent> {
        let elements = checked_element_count(desc)?;
        let byte_len = checked_byte_len(desc, elements)?;
        let start = self
            .data_region_start
            .checked_add(desc.data_offset)
            .ok_or_else(|| {
                GgufSnafu {
                    offset: self.data_region_start,
                    msg: format!(
                        "tensor `{}` start offset overflows u64: region_start={} + data_offset={}",
                        desc.name, self.data_region_start, desc.data_offset
                    ),
                }
                .build()
            })?;
        let end = start.checked_add(byte_len).ok_or_else(|| {
            GgufSnafu {
                offset: start,
                msg: format!(
                    "tensor `{}` end offset overflows u64: start={start} + byte_count={byte_len}",
                    desc.name
                ),
            }
            .build()
        })?;
        let mmap_len = u64::try_from(self.mmap.len()).map_err(|_| {
            GgufSnafu {
                offset: start,
                msg: format!("GGUF mmap length {} exceeds u64::MAX", self.mmap.len()),
            }
            .build()
        })?;
        if end > mmap_len {
            return GgufSnafu {
                offset: start,
                msg: format!(
                    "tensor `{}` data out of file bounds: [{start}..{end}] vs file len {mmap_len}",
                    desc.name
                ),
            }
            .fail();
        }
        Ok(TensorExtent {
            elements,
            start,
            end,
            byte_len,
        })
    }

    /// Compute a tensor's `[start, end)` byte range inside the mmap'd file
    /// and convert both ends to `usize`.
    fn byte_range_for(&self, desc: &TensorDescriptor) -> Result<(usize, usize)> {
        let extent = self.extent_for(desc)?;
        let start = usize::try_from(extent.start).map_err(|_| {
            GgufSnafu {
                offset: extent.start,
                msg: format!(
                    "tensor `{}` start offset {} exceeds usize::MAX",
                    desc.name, extent.start
                ),
            }
            .build()
        })?;
        let end = usize::try_from(extent.end).map_err(|_| {
            GgufSnafu {
                offset: extent.start,
                msg: format!(
                    "tensor `{}` end offset {} exceeds usize::MAX",
                    desc.name, extent.end
                ),
            }
            .build()
        })?;
        Ok((start, end))
    }

    fn validate_tensor_extents(&self) -> Result<()> {
        let mut extents = Vec::new();
        for desc in &self.tensors {
            let extent = self.extent_for(desc)?;
            extents.push((extent.start, extent.end, desc.name.as_str()));
        }
        extents.sort_unstable_by_key(|(start, _, _)| *start);
        for pair in extents.windows(2) {
            let (_, previous_end, previous_name) = pair[0];
            let (next_start, _, next_name) = pair[1];
            if next_start < previous_end {
                return GgufSnafu {
                    offset: next_start,
                    msg: format!(
                        "tensor `{next_name}` overlaps tensor `{previous_name}`: start {next_start} before prior end {previous_end}"
                    ),
                }
                .fail();
            }
        }
        Ok(())
    }
}

impl WeightProvider for Reader {
    fn get(&self, name: &str) -> Result<TensorView<'_>> {
        check_mmap_not_truncated(&self.path, self.mmap.len())?;
        let desc = self.descriptor_by_name(name)?;
        let dtype = desc.ggml_type.to_taxis_dtype()?;
        let (start_usize, end_usize) = self.byte_range_for(desc)?;
        let bytes = self.mmap.get(start_usize..end_usize).ok_or_else(|| {
            GgufSnafu {
                // u64::try_from on a freshly-`usize::try_from`'d value on any
                // 64-bit target is infallible; fall back to 0 on the
                // pathological 128-bit-usize case rather than propagating.
                offset: u64::try_from(start_usize).unwrap_or(0),
                msg: format!(
                    "tensor `{}` data slice [{start_usize}..{end_usize}] out of mmap",
                    desc.name
                ),
            }
            .build()
        })?;

        let shape = desc
            .dims
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| {
                    GgufSnafu {
                        offset: 0u64,
                        msg: format!("tensor `{}` dim {d} exceeds usize::MAX", desc.name),
                    }
                    .build()
                })
            })
            .collect::<Result<Vec<usize>>>()?;
        // Tie the borrow to &self (== mmap lifetime) cleanly.
        Ok(TensorView {
            name: &desc.name,
            dtype,
            shape,
            bytes,
        })
    }

    fn names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }

    fn len(&self) -> usize {
        self.tensors.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct TensorExtent {
    elements: u64,
    start: u64,
    end: u64,
    byte_len: u64,
}

fn checked_element_count(desc: &TensorDescriptor) -> Result<u64> {
    desc.dims.iter().try_fold(1u64, |elements, dim| {
        elements.checked_mul(*dim).ok_or_else(|| {
            GgufSnafu {
                offset: desc.data_offset,
                msg: format!(
                    "tensor `{}` dims product overflows u64: {:?}",
                    desc.name, desc.dims
                ),
            }
            .build()
        })
    })
}

fn checked_byte_len(desc: &TensorDescriptor, elements: u64) -> Result<u64> {
    if let Some((block_elements, block_bytes)) = desc.ggml_type.block_layout() {
        if !elements.is_multiple_of(block_elements) {
            return GgufSnafu {
                offset: desc.data_offset,
                msg: format!(
                    "tensor `{}` has {elements} elements, not a multiple of {block_elements} for {:?}",
                    desc.name, desc.ggml_type
                ),
            }
            .fail();
        }
        return (elements / block_elements)
            .checked_mul(block_bytes)
            .ok_or_else(|| {
                GgufSnafu {
                    offset: desc.data_offset,
                    msg: format!(
                        "tensor `{}` block-quant byte count overflows u64 for {:?}",
                        desc.name, desc.ggml_type
                    ),
                }
                .build()
            });
    }
    let bits = u64::try_from(desc.ggml_type.size_in_bits().ok_or_else(|| {
        GgufSnafu {
            offset: desc.data_offset,
            msg: format!("tensor `{}` has no GGML storage layout", desc.name),
        }
        .build()
    })?)
    .map_err(|_| {
        GgufSnafu {
            offset: desc.data_offset,
            msg: format!("tensor `{}` bit width exceeds u64::MAX", desc.name),
        }
        .build()
    })?;
    elements
        .checked_mul(bits)
        .ok_or_else(|| {
            GgufSnafu {
                offset: desc.data_offset,
                msg: format!(
                    "tensor `{}` byte count overflows u64: {bits} bits * {elements} elements",
                    desc.name
                ),
            }
            .build()
        })
        .map(|bits| bits.div_ceil(8))
}

fn validate_header_counts(cur: &Cursor<'_>, tensor_count: u64, metadata_count: u64) -> Result<()> {
    if tensor_count > MAX_TENSOR_COUNT {
        return GgufSnafu {
            offset: cur.pos,
            msg: format!("tensor count {tensor_count} exceeds limit {MAX_TENSOR_COUNT}"),
        }
        .fail();
    }
    if metadata_count > MAX_METADATA_COUNT {
        return GgufSnafu {
            offset: cur.pos,
            msg: format!("metadata count {metadata_count} exceeds limit {MAX_METADATA_COUNT}"),
        }
        .fail();
    }
    let required = metadata_count
        .checked_mul(MIN_METADATA_ENTRY_BYTES)
        .and_then(|metadata_bytes| {
            tensor_count
                .checked_mul(MIN_TENSOR_DESCRIPTOR_BYTES)
                .and_then(|tensor_bytes| metadata_bytes.checked_add(tensor_bytes))
        })
        .ok_or_else(|| {
            GgufSnafu {
                offset: cur.pos,
                msg: "GGUF header minimum size overflows u64".to_string(),
            }
            .build()
        })?;
    let remaining = cur.remaining_len()?;
    if required > remaining {
        return GgufSnafu {
            offset: cur.pos,
            msg: format!(
                "GGUF header declares {tensor_count} tensors and {metadata_count} metadata entries requiring at least {required} bytes, only {remaining} remain"
            ),
        }
        .fail();
    }
    Ok(())
}

fn validate_alignment(alignment: u64, offset: u64) -> Result<()> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return GgufSnafu {
            offset,
            msg: format!("GGUF alignment {alignment} is not a non-zero power of two"),
        }
        .fail();
    }
    Ok(())
}

fn align_up(offset: u64, alignment: u64) -> Result<u64> {
    validate_alignment(alignment, offset)?;
    let adjustment = alignment.checked_sub(1).ok_or_else(|| {
        GgufSnafu {
            offset,
            msg: "GGUF alignment underflows while rounding header".to_string(),
        }
        .build()
    })?;
    offset
        .checked_add(adjustment)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| {
            GgufSnafu {
                offset,
                msg: format!("GGUF header alignment overflows: {offset} rounded to {alignment}"),
            }
            .build()
        })
}

fn metadata_string(metadata: &HashMap<String, MetaValue>, key: &str) -> Option<String> {
    let MetaValue::String(value) = metadata.get(key)? else {
        return None;
    };
    Some(value.clone())
}

fn metadata_u32(metadata: &HashMap<String, MetaValue>, key: &str) -> Option<u32> {
    let MetaValue::U32(value) = metadata.get(key)? else {
        return None;
    };
    Some(*value)
}

// WHY(forkwright/logismos#60): `HashMap::insert` silently returns and
// drops the previous value. For a KV table or name index parsed from
// an untrusted file that makes duplicate keys/names a shadowing
// primitive: a crafted GGUF can declare a benign entry, then redeclare
// it later with different content, and only the second survives with
// nothing to say so. Both checks below reject the duplicate instead.

/// Reject a metadata key already present in `metadata`.
fn reject_duplicate_metadata_key(
    metadata: &HashMap<String, MetaValue>,
    key: &str,
    offset: u64,
) -> Result<()> {
    if metadata.contains_key(key) {
        return GgufSnafu {
            offset,
            msg: format!("duplicate metadata key `{key}`"),
        }
        .fail();
    }
    Ok(())
}

/// Reject a tensor name already present in `tensor_by_name`.
fn reject_duplicate_tensor_name(
    tensor_by_name: &HashMap<String, usize>,
    name: &str,
    offset: u64,
) -> Result<()> {
    if tensor_by_name.contains_key(name) {
        return GgufSnafu {
            offset,
            msg: format!("duplicate tensor name `{name}`"),
        }
        .fail();
    }
    Ok(())
}

/// Reject a tensor `dims` vector containing a zero-length dimension.
///
/// WHY(forkwright/logismos#60): a dims entry of 0 makes the
/// element-count product 0 regardless of the other dims, so the tensor
/// passes every later bounds check and yields an empty byte slice
/// while still reporting the original (non-empty-looking) `dims` in
/// its shape. Reject it here instead of letting it silently degrade to
/// a zero-byte tensor downstream.
fn reject_zero_length_dimension(name: &str, dims: &[u64], offset: u64) -> Result<()> {
    if dims.contains(&0) {
        return GgufSnafu {
            offset,
            msg: format!("tensor `{name}` has a zero-length dimension: {dims:?}"),
        }
        .fail();
    }
    Ok(())
}

/// Tiny stream-cursor over a mmap.
struct Cursor<'a> {
    mmap: &'a [u8],
    pos: u64,
    total_string_bytes: u64,
}

impl<'a> Cursor<'a> {
    fn new(mmap: &'a [u8]) -> Self {
        Self {
            mmap,
            pos: 0,
            total_string_bytes: 0,
        }
    }

    fn remaining_len(&self) -> Result<u64> {
        let mmap_len = u64::try_from(self.mmap.len()).map_err(|_| {
            GgufSnafu {
                offset: self.pos,
                msg: format!("GGUF mmap length {} exceeds u64::MAX", self.mmap.len()),
            }
            .build()
        })?;
        mmap_len.checked_sub(self.pos).ok_or_else(|| {
            GgufSnafu {
                offset: self.pos,
                msg: "GGUF cursor exceeds mmap length".to_string(),
            }
            .build()
        })
    }

    fn check_magic(&mut self) -> Result<()> {
        let magic = self.read_n(4)?;
        if magic != GGUF_MAGIC {
            let prefix = magic.get(..magic.len().min(4)).unwrap_or(magic);
            return GgufSnafu {
                offset: 0u64,
                msg: format!("bad magic: expected {GGUF_MAGIC:?}, got {prefix:?}"),
            }
            .fail();
        }
        Ok(())
    }

    fn read_n(&mut self, n: usize) -> Result<&'a [u8]> {
        let pos_usize = usize::try_from(self.pos).map_err(|_| {
            GgufSnafu {
                offset: self.pos,
                msg: format!("cursor pos {} exceeds usize::MAX", self.pos),
            }
            .build()
        })?;
        let end = pos_usize.checked_add(n).ok_or_else(|| {
            GgufSnafu {
                offset: self.pos,
                msg: format!("out-of-bounds read of {n}B at {}", self.pos),
            }
            .build()
        })?;
        let out = self.mmap.get(pos_usize..end).ok_or_else(|| {
            GgufSnafu {
                offset: self.pos,
                msg: format!("out-of-bounds read of {n}B at {}", self.pos),
            }
            .build()
        })?;
        self.pos = u64::try_from(end).map_err(|_| {
            GgufSnafu {
                offset: self.pos,
                msg: format!("cursor end {end} exceeds u64::MAX"),
            }
            .build()
        })?;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_n(1)?;
        let first = b.first().copied().ok_or_else(|| {
            GgufSnafu {
                offset: self.pos,
                msg: "short read of u8".to_string(),
            }
            .build()
        })?;
        Ok(first)
    }
    fn read_i8(&mut self) -> Result<i8> {
        let byte = self.read_u8()?;
        Ok(byte.cast_signed())
    }
    fn read_u16(&mut self) -> Result<u16> {
        let b: [u8; 2] = self.read_array()?;
        Ok(u16::from_le_bytes(b))
    }
    fn read_i16(&mut self) -> Result<i16> {
        let b: [u8; 2] = self.read_array()?;
        Ok(i16::from_le_bytes(b))
    }
    fn read_u32(&mut self) -> Result<u32> {
        let b: [u8; 4] = self.read_array()?;
        Ok(u32::from_le_bytes(b))
    }
    fn read_i32(&mut self) -> Result<i32> {
        let b: [u8; 4] = self.read_array()?;
        Ok(i32::from_le_bytes(b))
    }
    fn read_u64(&mut self) -> Result<u64> {
        let b: [u8; 8] = self.read_array()?;
        Ok(u64::from_le_bytes(b))
    }
    fn read_i64(&mut self) -> Result<i64> {
        let b: [u8; 8] = self.read_array()?;
        Ok(i64::from_le_bytes(b))
    }
    fn read_f32(&mut self) -> Result<f32> {
        let b: [u8; 4] = self.read_array()?;
        Ok(f32::from_le_bytes(b))
    }
    fn read_f64(&mut self) -> Result<f64> {
        let b: [u8; 8] = self.read_array()?;
        Ok(f64::from_le_bytes(b))
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let slice = self.read_n(N)?;
        let mut out = [0u8; N];
        if slice.len() != N {
            return GgufSnafu {
                offset: self.pos,
                msg: format!("short read of {N}B"),
            }
            .fail();
        }
        out.copy_from_slice(slice);
        Ok(out)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > MAX_STRING_BYTES {
            return GgufSnafu {
                offset: self.pos,
                msg: format!("GGUF string length {len} exceeds limit {MAX_STRING_BYTES}"),
            }
            .fail();
        }
        let total_string_bytes = self.total_string_bytes.checked_add(len).ok_or_else(|| {
            GgufSnafu {
                offset: self.pos,
                msg: "GGUF cumulative string bytes overflow u64".to_string(),
            }
            .build()
        })?;
        if total_string_bytes > MAX_TOTAL_STRING_BYTES {
            return GgufSnafu {
                offset: self.pos,
                msg: format!(
                    "GGUF cumulative string bytes {total_string_bytes} exceed limit {MAX_TOTAL_STRING_BYTES}"
                ),
            }
            .fail();
        }
        let len_usize = usize::try_from(len).map_err(|_| {
            GgufSnafu {
                offset: self.pos,
                msg: format!("gguf string length {len} exceeds usize::MAX"),
            }
            .build()
        })?;
        let bytes = self.read_n(len_usize)?;
        let string = std::str::from_utf8(bytes)
            .map(ToString::to_string)
            .map_err(|e| {
                GgufSnafu {
                    offset: self.pos - len,
                    msg: format!("invalid utf-8 in gguf string: {e}"),
                }
                .build()
            })?;
        self.total_string_bytes = total_string_bytes;
        Ok(string)
    }

    fn read_meta_value(&mut self) -> Result<MetaValue> {
        let type_id = self.read_u32()?;
        self.read_meta_value_typed(type_id)
    }

    fn read_meta_value_typed(&mut self, type_id: u32) -> Result<MetaValue> {
        Ok(match type_id {
            0 => MetaValue::U8(self.read_u8()?),
            1 => MetaValue::I8(self.read_i8()?),
            2 => MetaValue::U16(self.read_u16()?),
            3 => MetaValue::I16(self.read_i16()?),
            4 => MetaValue::U32(self.read_u32()?),
            5 => MetaValue::I32(self.read_i32()?),
            6 => MetaValue::F32(self.read_f32()?),
            7 => MetaValue::Bool(self.read_bool()?),
            8 => MetaValue::String(self.read_string()?),
            9 => {
                let inner_type = self.read_u32()?;
                // WHY(forkwright/logismos#35): the GGUF v3 spec forbids
                // arrays-of-arrays. Without this check a crafted file
                // chains `inner_type = 9` at every nesting level, and
                // each recursive `read_meta_value_typed` call below adds
                // an unbounded stack frame — a ~1MB file can encode
                // ~65,000 levels, enough to overflow the thread stack
                // (an unrecoverable process crash, not a catchable panic).
                if inner_type == 9 {
                    return GgufSnafu {
                        offset: self.pos,
                        msg: "gguf spec forbids arrays-of-arrays (inner_type=9)".to_string(),
                    }
                    .fail();
                }
                let n = self.read_u64()?;
                if n > MAX_METADATA_ARRAY_ELEMENTS {
                    return GgufSnafu {
                        offset: self.pos,
                        msg: format!(
                            "GGUF metadata array length {n} exceeds limit {MAX_METADATA_ARRAY_ELEMENTS}"
                        ),
                    }
                    .fail();
                }
                // WHY(forkwright/logismos#34): no pre-allocation from an
                // untrusted length — see the tensor_count/n_dims WHY
                // above in `Reader::open`. `Vec::push` amortises growth,
                // and each element still costs a real bounds-checked
                // read via `read_meta_value_typed`, so a claimed length
                // that outruns the file fails fast with `Error::Gguf`.
                let mut out = Vec::new();
                for _ in 0..n {
                    out.push(self.read_meta_value_typed(inner_type)?);
                }
                MetaValue::Array(out)
            }
            10 => MetaValue::U64(self.read_u64()?),
            11 => MetaValue::I64(self.read_i64()?),
            12 => MetaValue::F64(self.read_f64()?),
            other => {
                return GgufSnafu {
                    offset: self.pos,
                    msg: format!("unknown metadata type id {other}"),
                }
                .fail();
            }
        })
    }
}

#[cfg(test)]
#[path = "gguf_tests.rs"]
pub(crate) mod tests;
