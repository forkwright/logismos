//! Verified-payload row projection for the narrow Qwen3.5 boundary.
//!
//! WHY: structural preflight validates metadata and descriptor topology without
//! reading tensor bytes. This layer intentionally adds one digest-verified
//! payload-backed CPU operation without turning that cheap preflight into a
//! decoder-execution or model-support claim.

use snafu::ResultExt;

use loader::gguf::{GgmlType, VerifiedArtifact};
use quant::{RowFormat, row_byte_len, row_dot_f32};

use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, PayloadTensorSnafu, ProjectionAllocationSnafu, ProjectionBytesSnafu,
    ProjectionDtypeSnafu, ProjectionInputWidthSnafu, ProjectionLayoutSnafu, ProjectionRankSnafu,
    ProjectionRowSnafu,
};
use crate::qwen35::{Qwen35RecurrentLayout, Qwen35StructuralProfile};
use crate::qwen35_recurrent::Qwen35RecurrentExecution;

/// One payload-verified Qwen3.5 structural profile with a narrow CPU projection.
///
/// The only construction path borrows a [`VerifiedArtifact`], then repeats the
/// observation-only Qwen3.5 structural validation over its exact observation.
/// It accepts no external report as authority and exposes no constructor from
/// [`loader::gguf::ObservedArtifact`]. A successful construction is not a
/// full-decoder, logits, runtime-residency, or model-support claim.
///
/// ```compile_fail
/// use decoders::Qwen35Weights;
/// use loader::gguf::ObservedArtifact;
///
/// fn forge_from_observation(observed: &ObservedArtifact) {
///     let _ = Qwen35Weights::try_from_verified(observed);
/// }
/// ```
#[derive(Debug)]
pub struct Qwen35Weights<'artifact> {
    payload: &'artifact VerifiedArtifact,
    recurrent_layout: Qwen35RecurrentLayout,
}

impl<'artifact> Qwen35Weights<'artifact> {
    /// Bind one verified payload to the existing Qwen3.5 structural preflight.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the verified payload's retained
    /// observation does not meet the narrow Qwen3.5 structural contract.
    pub fn try_from_verified(payload: &'artifact VerifiedArtifact) -> Result<Self> {
        let profile = Qwen35StructuralProfile::try_from_observed(payload.observation())?;
        Ok(Self {
            payload,
            recurrent_layout: profile.recurrent_layout(),
        })
    }

    pub(crate) const fn recurrent_layout(&self) -> Qwen35RecurrentLayout {
        self.recurrent_layout
    }

    pub(crate) const fn payload(&self) -> &'artifact VerifiedArtifact {
        self.payload
    }

    /// Project finite activations through one named, recognized executable matrix.
    ///
    /// GGUF matrix dimensions are `[input_width, output_width]`; each output
    /// row therefore occupies one contiguous complete serialized row. All tensor
    /// geometry is checked before output allocation. A row failure drops the
    /// local vector, so this method never returns a partial projection.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] for an absent tensor, an unsupported matrix
    /// dtype/rank/geometry, allocation failure, or invalid serialized-row
    /// arithmetic.
    pub fn project(&self, name: &str, activations: &[f32]) -> Result<Vec<f32>> {
        let tensor = self.payload.tensor(name).context(PayloadTensorSnafu {
            name: name.to_string(),
        })?;
        let tensor_name = tensor.name().to_string();
        let dims = tensor.dims();
        let [input_width, output_width] = dims else {
            return ProjectionRankSnafu {
                name: tensor_name,
                actual: dims.len(),
            }
            .fail();
        };
        let format = row_format(tensor.ggml_type()).ok_or_else(|| {
            ProjectionDtypeSnafu {
                name: tensor_name.clone(),
                actual: tensor.ggml_type(),
            }
            .build()
        })?;
        let input_width = usize::try_from(*input_width).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "row projection input width",
            }
            .build()
        })?;
        let output_width = usize::try_from(*output_width).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "row projection output width",
            }
            .build()
        })?;
        if activations.len() != input_width {
            return ProjectionInputWidthSnafu {
                name: tensor_name,
                expected: input_width,
                actual: activations.len(),
            }
            .fail();
        }
        let row_byte_len =
            row_byte_len(format, input_width).with_context(|_| ProjectionLayoutSnafu {
                name: tensor_name.clone(),
            })?;
        let expected_bytes = output_width.checked_mul(row_byte_len).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "row projection payload byte length",
            }
            .build()
        })?;
        let bytes = tensor.bytes();
        if bytes.len() != expected_bytes {
            return ProjectionBytesSnafu {
                name: tensor_name,
                expected: expected_bytes,
                actual: bytes.len(),
            }
            .fail();
        }

        let mut output = Vec::new();
        output
            .try_reserve_exact(output_width)
            .with_context(|_| ProjectionAllocationSnafu {
                name: tensor_name.clone(),
                output_width,
            })?;
        for (row, row_bytes) in bytes.chunks_exact(row_byte_len).enumerate() {
            let value = row_dot_f32(format, row_bytes, activations).with_context(|_| {
                ProjectionRowSnafu {
                    name: tensor_name.clone(),
                    row,
                }
            })?;
            output.push(value);
        }
        Ok(output)
    }

    /// Prepare one stateful recurrent-attention trunk for a recurrent main block.
    ///
    /// WHY: state construction remains bound to this digest-verified payload and
    /// one checked block role inventory, so callers cannot pair an arbitrary
    /// recurrence state with a similarly shaped artifact.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the selected block is not recurrent or
    /// when its execution-only finite parameters cannot be admitted.
    pub fn recurrent_execution(
        &self,
        block_index: u64,
    ) -> Result<Qwen35RecurrentExecution<'_, 'artifact>> {
        Qwen35RecurrentExecution::try_from_weights(self, block_index)
    }
}

fn row_format(ggml_type: GgmlType) -> Option<RowFormat> {
    match ggml_type {
        GgmlType::F32 => Some(RowFormat::F32),
        GgmlType::Q8_0 => Some(RowFormat::Q8_0),
        GgmlType::Q4K => Some(RowFormat::Q4K),
        GgmlType::Q5K => Some(RowFormat::Q5K),
        GgmlType::Q6K => Some(RowFormat::Q6K),
        _ => None,
    }
}
