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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::{TensorView, WeightProvider, check_mmap_not_truncated};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_V3: u32 = 3;

/// ggml / GGUF dtype tag. Numeric ids match the spec verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    fn from_u32(id: u32) -> Result<Self> {
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
                return Err(Error::Gguf {
                    offset: 0,
                    msg: format!("unknown ggml type id {other}"),
                });
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

    fn to_taxis_dtype(self) -> Result<taxis::DType> {
        Ok(match self {
            Self::F32 => taxis::DType::F32,
            Self::F16 => taxis::DType::F16,
            Self::BF16 => taxis::DType::BF16,
            Self::I32 => taxis::DType::I32,
            Self::I8 => taxis::DType::I8,
            other => {
                return Err(Error::Gguf {
                    offset: 0,
                    msg: format!(
                        "ggml type {other:?} not decodable in Phase 2 \
                         (K-quant / block dtypes land with quant kernels in Phase 6)"
                    ),
                });
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
    /// malformed header, an unsupported version, an out-of-range count,
    /// a duplicate metadata key, a duplicate tensor name, or a
    /// zero-length tensor dimension.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: see `safetensors::Reader::open`.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        let mut cur = Cursor::new(&mmap);
        cur.check_magic()?;
        let version = cur.read_u32()?;
        if version != GGUF_V3 {
            return Err(Error::Gguf {
                offset: cur.pos,
                msg: format!("unsupported GGUF version {version} (target: v{GGUF_V3})"),
            });
        }
        let tensor_count = cur.read_u64()?;
        let metadata_count = cur.read_u64()?;

        let mut metadata: HashMap<String, MetaValue> = HashMap::new();
        for _ in 0..metadata_count {
            let key = cur.read_string()?;
            let val = cur.read_meta_value()?;
            // PROOF-BRANCH: duplicate-key check reverted to demonstrate
            // `rejects_duplicate_metadata_key` fails against it. Not for merge.
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(MetaValue::U32(v)) => u64::from(*v),
            _ => 32,
        };

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
        usize::try_from(tensor_count).map_err(|_| Error::Gguf {
            offset: cur.pos,
            msg: format!("tensor count {tensor_count} exceeds usize::MAX"),
        })?;
        let mut tensors = Vec::new();
        let mut tensor_by_name = HashMap::new();
        for i in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()?;
            // WHY(forkwright/logismos#34): same rationale as tensor_count
            // above — no pre-allocation from an untrusted count.
            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            // PROOF-BRANCH: zero-dim check reverted to demonstrate
            // `rejects_zero_length_dimension` fails against it. Not for merge.
            let ggml_type_id = cur.read_u32()?;
            let ggml_type = GgmlType::from_u32(ggml_type_id)?;
            let data_offset = cur.read_u64()?;
            let idx_usize = usize::try_from(i).map_err(|_| Error::Gguf {
                offset: cur.pos,
                msg: format!("tensor index {i} exceeds usize::MAX"),
            })?;
            // PROOF-BRANCH: duplicate-name check reverted to demonstrate
            // `rejects_duplicate_tensor_name` fails against it. Not for merge.
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
        let data_region_start = align_up(after_header, alignment);

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            metadata,
            tensors,
            tensor_by_name,
            data_region_start,
            alignment,
        })
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
}

impl Reader {
    /// Look up a tensor descriptor by name.
    fn descriptor_by_name(&self, name: &str) -> Result<&TensorDescriptor> {
        let idx = self
            .tensor_by_name
            .get(name)
            .copied()
            .ok_or_else(|| Error::TensorNotFound {
                name: name.to_string(),
            })?;
        self.tensors.get(idx).ok_or_else(|| Error::Gguf {
            offset: 0,
            msg: format!(
                "internal: tensor index {idx} out of range (len={})",
                self.tensors.len()
            ),
        })
    }

    /// Compute a tensor's `[start, end)` byte range inside the mmap'd file
    /// and convert both ends to `usize`.
    fn byte_range_for(&self, desc: &TensorDescriptor) -> Result<(usize, usize)> {
        // WHY(forkwright/logismos#36): `Iterator::product()` on `u64`
        // uses ordinary wrapping multiplication in a release build — a
        // GGUF-supplied `dims` whose product exceeds `u64::MAX` silently
        // wraps, and can land on exactly `0`. A zero element count then
        // produces a zero-byte `TensorView` that still reports the
        // original (enormous) `dims` in its `shape`, passing every
        // bounds check unchanged instead of failing loudly. `checked_mul`
        // turns that silent wraparound into a returned `Err`.
        let mut elem_count_u64: u64 = 1;
        for &d in &desc.dims {
            elem_count_u64 = elem_count_u64.checked_mul(d).ok_or_else(|| Error::Gguf {
                offset: 0,
                msg: format!(
                    "tensor `{}` dims product overflows u64: {:?}",
                    desc.name, desc.dims
                ),
            })?;
        }
        let elem_count = usize::try_from(elem_count_u64).map_err(|_| Error::Gguf {
            offset: 0,
            msg: format!(
                "tensor `{}` element count {elem_count_u64} exceeds usize::MAX",
                desc.name
            ),
        })?;
        let bits = desc.ggml_type.size_in_bits().ok_or_else(|| Error::Gguf {
            offset: 0,
            msg: format!(
                "tensor `{}` uses block-quant dtype {:?}; block decoding is Phase 6",
                desc.name, desc.ggml_type
            ),
        })?;
        // PROOF-BRANCH: checked_mul reverted to saturating_mul to
        // demonstrate `byte_range_for_rejects_byte_count_overflow`
        // fails against it. Not for merge.
        let byte_count = bits.saturating_mul(elem_count).div_ceil(8);
        let byte_count_u64 = u64::try_from(byte_count).map_err(|_| Error::Gguf {
            offset: 0,
            msg: format!(
                "tensor `{}` byte count {byte_count} exceeds u64::MAX",
                desc.name
            ),
        })?;

        // WHY(forkwright/logismos#56): both additions are on untrusted
        // u64s straight off the wire (the file-supplied `data_offset`,
        // and a byte count derived from file-supplied dims/dtype) and
        // wrap silently in a release build, which can land `start`/`end`
        // on an arbitrary in-bounds-looking mmap offset instead of
        // failing loudly. `checked_add` turns the wrap into `Err`.
        let start = self
            .data_region_start
            .checked_add(desc.data_offset)
            .ok_or_else(|| Error::Gguf {
                offset: 0,
                msg: format!(
                    "tensor `{}` start offset overflows u64: region_start={} + data_offset={}",
                    desc.name, self.data_region_start, desc.data_offset
                ),
            })?;
        let end = start.checked_add(byte_count_u64).ok_or_else(|| Error::Gguf {
            offset: start,
            msg: format!(
                "tensor `{}` end offset overflows u64: start={start} + byte_count={byte_count_u64}",
                desc.name
            ),
        })?;
        let start_usize = usize::try_from(start).map_err(|_| Error::Gguf {
            offset: start,
            msg: format!(
                "tensor `{}` start offset {start} exceeds usize::MAX",
                desc.name
            ),
        })?;
        let end_usize = usize::try_from(end).map_err(|_| Error::Gguf {
            offset: start,
            msg: format!("tensor `{}` end offset {end} exceeds usize::MAX", desc.name),
        })?;
        if end_usize > self.mmap.len() {
            return Err(Error::Gguf {
                offset: start,
                msg: format!(
                    "tensor `{}` data out of file bounds: [{start}..{end}] vs file len {}",
                    desc.name,
                    self.mmap.len()
                ),
            });
        }
        Ok((start_usize, end_usize))
    }
}

impl WeightProvider for Reader {
    fn get(&self, name: &str) -> Result<TensorView<'_>> {
        check_mmap_not_truncated(&self.path, self.mmap.len())?;
        let desc = self.descriptor_by_name(name)?;
        let dtype = desc.ggml_type.to_taxis_dtype()?;
        let (start_usize, end_usize) = self.byte_range_for(desc)?;
        let bytes = self
            .mmap
            .get(start_usize..end_usize)
            .ok_or_else(|| Error::Gguf {
                // u64::try_from on a freshly-`usize::try_from`'d value on any
                // 64-bit target is infallible; fall back to 0 on the
                // pathological 128-bit-usize case rather than propagating.
                offset: u64::try_from(start_usize).unwrap_or(0),
                msg: format!(
                    "tensor `{}` data slice [{start_usize}..{end_usize}] out of mmap",
                    desc.name
                ),
            })?;

        let shape = desc
            .dims
            .iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| Error::Gguf {
                    offset: 0,
                    msg: format!("tensor `{}` dim {d} exceeds usize::MAX", desc.name),
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

fn align_up(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    offset.div_ceil(alignment) * alignment
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
        return Err(Error::Gguf {
            offset,
            msg: format!("duplicate metadata key `{key}`"),
        });
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
        return Err(Error::Gguf {
            offset,
            msg: format!("duplicate tensor name `{name}`"),
        });
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
        return Err(Error::Gguf {
            offset,
            msg: format!("tensor `{name}` has a zero-length dimension: {dims:?}"),
        });
    }
    Ok(())
}

/// Tiny stream-cursor over a mmap.
struct Cursor<'a> {
    mmap: &'a [u8],
    pos: u64,
}

impl<'a> Cursor<'a> {
    fn new(mmap: &'a [u8]) -> Self {
        Self { mmap, pos: 0 }
    }

    fn check_magic(&mut self) -> Result<()> {
        let magic = self.read_n(4)?;
        if magic != GGUF_MAGIC {
            let prefix = magic.get(..magic.len().min(4)).unwrap_or(magic);
            return Err(Error::Gguf {
                offset: 0,
                msg: format!("bad magic: expected {GGUF_MAGIC:?}, got {prefix:?}"),
            });
        }
        Ok(())
    }

    fn read_n(&mut self, n: usize) -> Result<&'a [u8]> {
        let pos_usize = usize::try_from(self.pos).map_err(|_| Error::Gguf {
            offset: self.pos,
            msg: format!("cursor pos {} exceeds usize::MAX", self.pos),
        })?;
        let end = pos_usize.checked_add(n).ok_or_else(|| Error::Gguf {
            offset: self.pos,
            msg: format!("out-of-bounds read of {n}B at {}", self.pos),
        })?;
        let out = self.mmap.get(pos_usize..end).ok_or_else(|| Error::Gguf {
            offset: self.pos,
            msg: format!("out-of-bounds read of {n}B at {}", self.pos),
        })?;
        self.pos = u64::try_from(end).map_err(|_| Error::Gguf {
            offset: self.pos,
            msg: format!("cursor end {end} exceeds u64::MAX"),
        })?;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_n(1)?;
        let first = b.first().copied().ok_or_else(|| Error::Gguf {
            offset: self.pos,
            msg: "short read of u8".into(),
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
            return Err(Error::Gguf {
                offset: self.pos,
                msg: format!("short read of {N}B"),
            });
        }
        out.copy_from_slice(slice);
        Ok(out)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        let len_usize = usize::try_from(len).map_err(|_| Error::Gguf {
            offset: self.pos,
            msg: format!("gguf string length {len} exceeds usize::MAX"),
        })?;
        let bytes = self.read_n(len_usize)?;
        std::str::from_utf8(bytes)
            .map(ToString::to_string)
            .map_err(|e| Error::Gguf {
                offset: self.pos - len,
                msg: format!("invalid utf-8 in gguf string: {e}"),
            })
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
                    return Err(Error::Gguf {
                        offset: self.pos,
                        msg: "gguf spec forbids arrays-of-arrays (inner_type=9)".into(),
                    });
                }
                let n = self.read_u64()?;
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
                return Err(Error::Gguf {
                    offset: self.pos,
                    msg: format!("unknown metadata type id {other}"),
                });
            }
        })
    }
}

#[cfg(test)]
#[path = "gguf_tests.rs"]
pub(crate) mod tests;
