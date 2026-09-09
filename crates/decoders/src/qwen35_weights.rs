//! Verified-payload row projection for the narrow Qwen3.5 boundary.
//!
//! WHY: structural preflight validates metadata and descriptor topology without
//! reading tensor bytes. This layer intentionally adds one digest-verified
//! payload-backed CPU operation without turning that cheap preflight into a
//! decoder-execution or model-support claim.

use loader::gguf::VerifiedArtifact;

use crate::Result;
use crate::matrix::CheckedMatrix;
use crate::qwen35::{Qwen35ExecutionDimensions, Qwen35RecurrentLayout, Qwen35StructuralProfile};
use crate::qwen35_execution::{Qwen35Execution, Qwen35ExecutionPlan, Qwen35LogitSelection};
use crate::qwen35_recurrent::Qwen35RecurrentExecution;

/// One payload-verified Qwen3.5 structural profile with a narrow CPU projection.
///
/// The only construction path borrows a [`VerifiedArtifact`], then retains a
/// cheap shared clone and repeats the observation-only Qwen3.5 structural
/// validation over its exact observation.
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
pub struct Qwen35Weights {
    payload: VerifiedArtifact,
    recurrent_layout: Qwen35RecurrentLayout,
    execution_dimensions: Qwen35ExecutionDimensions,
}

impl Qwen35Weights {
    pub(crate) const fn projection_output_elements(output_width: usize) -> usize {
        CheckedMatrix::projection_output_elements(output_width)
    }

    pub(crate) const fn decoded_row_elements(input_width: usize) -> usize {
        CheckedMatrix::decoded_row_elements(input_width)
    }

    /// Bind one verified payload to the existing Qwen3.5 structural preflight.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the verified payload's retained
    /// observation does not meet the narrow Qwen3.5 structural contract.
    pub fn try_from_verified(payload: &VerifiedArtifact) -> Result<Self> {
        let profile = Qwen35StructuralProfile::try_from_observed(payload.observation())?;
        Ok(Self {
            payload: payload.clone(),
            recurrent_layout: profile.recurrent_layout(),
            execution_dimensions: profile.execution_dimensions(),
        })
    }

    pub(crate) const fn recurrent_layout(&self) -> Qwen35RecurrentLayout {
        self.recurrent_layout
    }

    pub(crate) const fn payload(&self) -> &VerifiedArtifact {
        &self.payload
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
        self.matrix(name)?.project(activations)
    }

    /// Decode one checked matrix row without materialising every output row.
    ///
    /// This is used for token embedding lookup, whose GGUF storage is a
    /// matrix with vocabulary rows and hidden columns.
    pub(crate) fn decode_row(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        self.matrix(name)?.decode_row(row)
    }

    fn matrix(&self, name: &str) -> Result<CheckedMatrix<'_>> {
        CheckedMatrix::from_payload(self.payload, name)
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn checked_matrix(&self, name: &str) -> Result<CheckedMatrix<'_>> {
        self.matrix(name)
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
    pub fn recurrent_execution(&self, block_index: u64) -> Result<Qwen35RecurrentExecution<'_>> {
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
    pub fn execution(&self, max_context: usize) -> Result<Qwen35Execution<'_>> {
        Qwen35Execution::try_from_weights(self, max_context)
    }

    /// Derive one artifact-bound CPU execution plan before allocating session state.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the requested context or step bound cannot
    /// be admitted from this verified artifact's execution metadata.
    pub fn execution_plan(
        &self,
        max_context: usize,
        max_step_tokens: usize,
        selection: Qwen35LogitSelection,
    ) -> Result<Qwen35ExecutionPlan<'_>> {
        Qwen35ExecutionPlan::try_from_weights(self, max_context, max_step_tokens, selection)
    }
}
