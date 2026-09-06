//! Verified-payload row projection for the narrow Qwen3.5 boundary.
//!
//! WHY: structural preflight validates metadata and descriptor topology without
//! reading tensor bytes. This layer intentionally adds one digest-verified
//! payload-backed CPU operation without turning that cheap preflight into a
//! decoder-execution or model-support claim.

use snafu::ResultExt;

use loader::gguf::{GgmlType, VerifiedArtifact, VerifiedTensor};
use quant::{RowFormat, row_byte_len, row_decode_f32, row_dot_f32};

use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, PayloadTensorSnafu, ProjectionAllocationSnafu, ProjectionBytesSnafu,
    ProjectionDtypeSnafu, ProjectionInputWidthSnafu, ProjectionLayoutSnafu, ProjectionRankSnafu,
    ProjectionRowSnafu,
};
use crate::qwen35::{Qwen35ExecutionDimensions, Qwen35RecurrentLayout, Qwen35StructuralProfile};
use crate::qwen35_execution::Qwen35Execution;
use crate::qwen35_recurrent::Qwen35RecurrentExecution;

/// One payload-verified Qwen3.5 structural profile with a narrow CPU projection.
///
/// The only construction path borrows a [`VerifiedArtifact`], then repeats the
/// observation-only Qwen3.5 structural validation over its exact observation.
/// It accepts no external report as authority and exposes no constructor from
/// [`loader::gguf::ObservedArtifact`]. A successful binding establishes only
/// verified payload ownership and the structural profile; execution-only
/// metadata and arithmetic are validated separately by [`Self::execution`] and
/// [`Qwen35Execution::step`]. It does not establish
/// tokenizer/template, sampling, `NextN`, runtime-residency, GPU, or
/// model-support qualification.
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
    execution_dimensions: Qwen35ExecutionDimensions,
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
            execution_dimensions: profile.execution_dimensions(),
        })
    }

    pub(crate) const fn recurrent_layout(&self) -> Qwen35RecurrentLayout {
        self.recurrent_layout
    }

    pub(crate) const fn payload(&self) -> &'artifact VerifiedArtifact {
        self.payload
    }

    pub(crate) const fn execution_dimensions(&self) -> Qwen35ExecutionDimensions {
        self.execution_dimensions
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
        let matrix = self.matrix(name)?;
        if activations.len() != matrix.input_width {
            return ProjectionInputWidthSnafu {
                name: matrix.name,
                expected: matrix.input_width,
                actual: activations.len(),
            }
            .fail();
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(matrix.output_width)
            .with_context(|_| ProjectionAllocationSnafu {
                name: matrix.name.clone(),
                output_width: matrix.output_width,
            })?;
        for (row, row_bytes) in matrix
            .tensor
            .bytes()
            .chunks_exact(matrix.row_byte_len)
            .enumerate()
        {
            let value = row_dot_f32(matrix.format, row_bytes, activations).with_context(|_| {
                ProjectionRowSnafu {
                    name: matrix.name.clone(),
                    row,
                }
            })?;
            output.push(value);
        }
        Ok(output)
    }

    /// Decode one checked matrix row without materialising every output row.
    ///
    /// This is used for token embedding lookup, whose GGUF storage is a
    /// matrix with vocabulary rows and hidden columns.
    pub(crate) fn decode_row(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let matrix = self.matrix(name)?;
        let format = matrix.format;
        let input_width = matrix.input_width;
        let matrix_name = matrix.name.clone();
        let row_bytes = matrix.row(row)?;
        row_decode_f32(format, row_bytes, input_width).with_context(|_| ProjectionRowSnafu {
            name: matrix_name,
            row,
        })
    }

    fn matrix(&self, name: &str) -> Result<CheckedMatrix<'_>> {
        CheckedMatrix::from_payload(self.payload, name)
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

    /// Bind a bounded, stateful CPU text session to this verified payload.
    ///
    /// The caller's bound limits retained recurrent and KV state. It is an
    /// execution request, not a physical reservation or an artifact claim.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when execution-only Qwen3.5 metadata is
    /// incomplete, unsupported, or the requested context is out of range.
    pub fn execution(&self, max_context: usize) -> Result<Qwen35Execution<'_, 'artifact>> {
        Qwen35Execution::try_from_weights(self, max_context)
    }
}

struct CheckedMatrix<'a> {
    name: String,
    tensor: VerifiedTensor<'a>,
    format: RowFormat,
    input_width: usize,
    output_width: usize,
    row_byte_len: usize,
}

impl<'a> CheckedMatrix<'a> {
    fn from_payload(payload: &'a VerifiedArtifact, name: &str) -> Result<Self> {
        let tensor = payload.tensor(name).context(PayloadTensorSnafu {
            name: name.to_string(),
        })?;
        let tensor_name = tensor.name().to_string();
        let [input_width, output_width] = tensor.dims() else {
            return ProjectionRankSnafu {
                name: tensor_name,
                actual: tensor.dims().len(),
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
        Ok(Self {
            name: tensor.name().to_string(),
            tensor,
            format,
            input_width,
            output_width,
            row_byte_len,
        })
    }

    fn row(&self, row: usize) -> Result<&[u8]> {
        if row >= self.output_width {
            return ProjectionInputWidthSnafu {
                name: self.name.clone(),
                expected: self.output_width,
                actual: row,
            }
            .fail();
        }
        let start = row.checked_mul(self.row_byte_len).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "row projection row offset",
            }
            .build()
        })?;
        let end = start.checked_add(self.row_byte_len).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "row projection row end",
            }
            .build()
        })?;
        self.tensor.bytes().get(start..end).ok_or_else(|| {
            ProjectionBytesSnafu {
                name: self.name.clone(),
                expected: end,
                actual: self.tensor.bytes().len(),
            }
            .build()
        })
    }
}

fn row_format(ggml_type: GgmlType) -> Option<RowFormat> {
    match ggml_type {
        GgmlType::F32 => Some(RowFormat::F32),
        GgmlType::Q8_0 => Some(RowFormat::Q8_0),
        GgmlType::Q4K => Some(RowFormat::Q4K),
        GgmlType::Q5K => Some(RowFormat::Q5K),
        GgmlType::Q6K => Some(RowFormat::Q6K),
        GgmlType::IQ4NL => Some(RowFormat::IQ4NL),
        GgmlType::IQ4XS => Some(RowFormat::IQ4XS),
        _ => None,
    }
}
