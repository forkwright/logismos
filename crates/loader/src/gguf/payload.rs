//! Verified, privately owned GGUF payload backing.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;

use rustix::fs::{Mode, OFlags};
use sha2::{Digest, Sha256};
use snafu::ResultExt;

use super::{ArtifactDigest, ObservedArtifact, Reader, Sha256Digest, TensorDescriptor};
use crate::error::{
    ArtifactBackingAllocationSnafu, ArtifactDigestMismatchSnafu, ArtifactExceedsByteLimitSnafu,
    ArtifactInputNotRegularSnafu, GgufSnafu, Result,
};

/// Explicit maximum number of serialized artifact bytes held in one backing.
///
/// This limit applies only to the owned GGUF byte backing. Parser metadata,
/// descriptors, and indexes have their own bounded parsing policy and are not
/// a host-memory reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct ArtifactByteLimit(NonZeroU64);

impl ArtifactByteLimit {
    /// Construct a non-zero serialized-artifact backing limit.
    #[must_use]
    pub const fn new(serialized_bytes: NonZeroU64) -> Self {
        Self(serialized_bytes)
    }

    /// Return the maximum permitted serialized backing length.
    #[must_use]
    pub const fn get(self) -> NonZeroU64 {
        self.0
    }
}

/// Immutable GGUF bytes verified against a required SHA-256 digest.
///
/// Clones retain the same private backing and observation through shared
/// ownership; they do not reread `path` or duplicate serialized payload bytes.
/// The backing is read exactly once from `path`; later tensor access neither
/// maps nor reopens that path. Digest equality proves only equality to the
/// caller-supplied bytes: it does not establish publisher authenticity or an
/// atomic filesystem snapshot.
///
/// ```compile_fail
/// use loader::gguf::{ObservedArtifact, VerifiedArtifact};
///
/// fn forge_artifact(observation: ObservedArtifact) {
///     let _ = VerifiedArtifact::from_parts(Vec::new(), observation);
/// }
/// ```
#[derive(Clone)]
pub struct VerifiedArtifact {
    inner: Arc<VerifiedArtifactInner>,
}

struct VerifiedArtifactInner {
    backing: Vec<u8>,
    observation: ObservedArtifact,
}

impl VerifiedArtifact {
    /// Copy, verify, and parse one GGUF artifact into immutable process-owned bytes.
    ///
    /// The supplied digest is required because a digest computed only after a
    /// read identifies whatever bytes happened to be copied, rather than an
    /// intended artifact. The limit is explicit because this CPU backing is a
    /// deliberate host-memory commitment.
    ///
    /// The file length captured before the read fixes the byte range copied
    /// into this backing. A later append is deliberately neither re-read nor
    /// re-statted: it cannot alter the owned bytes or their identity. A rewrite
    /// within that captured range is admitted only if its copied bytes still
    /// equal `expected`; this API does not claim an atomic filesystem snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Io`] when the input cannot be opened or read;
    /// [`crate::Error::ArtifactExceedsByteLimit`] before allocating or reading
    /// payload bytes when the opened file is too large;
    /// [`crate::Error::ArtifactBackingAllocation`] when the backing cannot be
    /// reserved; [`crate::Error::ArtifactDigestMismatch`] when copied bytes do
    /// not match `expected`; or [`crate::Error::Gguf`] for malformed or
    /// overflowed GGUF content whose digest did match.
    pub fn load(path: &Path, expected: Sha256Digest, limit: ArtifactByteLimit) -> Result<Self> {
        let (mut file, serialized_bytes) = open_regular_file(path)?;
        let backing = read_backing(&mut file, serialized_bytes, limit)?;
        let actual = Sha256Digest::from_bytes(Sha256::digest(&backing).into());
        if actual != expected {
            return ArtifactDigestMismatchSnafu { expected, actual }.fail();
        }

        let parsed = Reader::parse(&backing)?;
        parsed.validate_tensor_extents(serialized_bytes)?;
        let inspection = parsed.inspection(serialized_bytes, ArtifactDigest::Sha256(actual))?;
        Ok(Self::from_parts(
            backing,
            ObservedArtifact { inspection, parsed },
        ))
    }

    /// Borrow reporting facts derived from the same owned and verified bytes.
    #[must_use]
    pub fn observation(&self) -> &ObservedArtifact {
        &self.inner.observation
    }

    /// Borrow one checked tensor from this artifact's private backing.
    ///
    /// The returned view carries only metadata and bytes selected by this
    /// artifact's validated name index and extent. Callers cannot substitute a
    /// foreign descriptor, report, or byte range.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::TensorNotFound`] when `name` does not occur in
    /// this verified artifact, or [`crate::Error::Gguf`] if an internal checked
    /// extent cannot be represented as a backing slice.
    pub fn tensor(&self, name: &str) -> Result<VerifiedTensor<'_>> {
        let descriptor = self.observation().descriptor_by_name(name)?;
        let serialized_bytes = u64::try_from(self.inner.backing.len()).map_err(|_| {
            GgufSnafu {
                offset: 0u64,
                msg: format!(
                    "verified artifact backing length {} exceeds u64::MAX",
                    self.inner.backing.len()
                ),
            }
            .build()
        })?;
        let extent = self
            .observation()
            .parsed
            .extent_for(descriptor, serialized_bytes)?;
        let start = usize::try_from(extent.start).map_err(|_| {
            GgufSnafu {
                offset: extent.start,
                msg: format!(
                    "tensor `{}` start offset {} exceeds usize::MAX",
                    descriptor.name, extent.start
                ),
            }
            .build()
        })?;
        let end = usize::try_from(extent.end).map_err(|_| {
            GgufSnafu {
                offset: extent.start,
                msg: format!(
                    "tensor `{}` end offset {} exceeds usize::MAX",
                    descriptor.name, extent.end
                ),
            }
            .build()
        })?;
        let bytes = self.inner.backing.get(start..end).ok_or_else(|| {
            GgufSnafu {
                offset: extent.start,
                msg: format!(
                    "tensor `{}` checked range [{start}..{end}] is outside verified backing",
                    descriptor.name
                ),
            }
            .build()
        })?;
        Ok(VerifiedTensor { descriptor, bytes })
    }

    fn from_parts(backing: Vec<u8>, observation: ObservedArtifact) -> Self {
        Self {
            inner: Arc::new(VerifiedArtifactInner {
                backing,
                observation,
            }),
        }
    }
}

/// Open and inspect exactly the descriptor that will provide artifact bytes.
///
/// `NONBLOCK` prevents a FIFO endpoint from making the verification path wait
/// for a writer. `NOFOLLOW` rejects a final-component symlink instead of
/// resolving it between a pathname check and open. Metadata is then queried on
/// the returned descriptor, so a pathname replacement cannot change its type
/// or length after this check.
fn open_regular_file(path: &Path) -> Result<(File, u64)> {
    let descriptor = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(std::io::Error::from)?;
    let file = File::from(descriptor);
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return ArtifactInputNotRegularSnafu {
            path: path.to_path_buf(),
        }
        .fail();
    }
    Ok((file, metadata.len()))
}

impl fmt::Debug for VerifiedArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedArtifact")
            .field("serialized_bytes", &self.inner.backing.len())
            .field("digest", &self.inner.observation.inspection().digest)
            .finish_non_exhaustive()
    }
}

/// A checked tensor view borrowed from one [`VerifiedArtifact`].
///
/// ```compile_fail
/// use loader::gguf::VerifiedTensor;
///
/// fn mutate_tensor(tensor: &mut VerifiedTensor<'_>) {
///     tensor.bytes = &[];
/// }
/// ```
pub struct VerifiedTensor<'artifact> {
    descriptor: &'artifact TensorDescriptor,
    bytes: &'artifact [u8],
}

impl<'artifact> VerifiedTensor<'artifact> {
    /// Return this verified tensor's source name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.descriptor.name
    }

    /// Return this verified tensor's checked GGML storage type.
    #[must_use]
    pub const fn ggml_type(&self) -> super::GgmlType {
        self.descriptor.ggml_type
    }

    /// Return this verified tensor's checked logical dimensions.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.descriptor.dims
    }

    /// Borrow this verified tensor's immutable serialized bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

impl fmt::Debug for VerifiedTensor<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedTensor")
            .field("name", &self.name())
            .field("ggml_type", &self.ggml_type())
            .field("dims", &self.dims())
            .field("serialized_bytes", &self.bytes.len())
            .finish()
    }
}

pub(super) fn read_backing(
    reader: &mut impl Read,
    serialized_bytes: u64,
    limit: ArtifactByteLimit,
) -> Result<Vec<u8>> {
    if serialized_bytes > limit.get().get() {
        return ArtifactExceedsByteLimitSnafu {
            serialized_bytes,
            limit: limit.get(),
        }
        .fail();
    }
    let backing_len = usize::try_from(serialized_bytes).map_err(|_| {
        GgufSnafu {
            offset: 0u64,
            msg: format!("artifact backing length {serialized_bytes} exceeds usize::MAX"),
        }
        .build()
    })?;
    let mut backing = Vec::new();
    backing
        .try_reserve_exact(backing_len)
        .context(ArtifactBackingAllocationSnafu { serialized_bytes })?;
    backing.resize(backing_len, 0u8);
    reader.read_exact(&mut backing)?;
    Ok(backing)
}
