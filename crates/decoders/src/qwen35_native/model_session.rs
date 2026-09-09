//! Blocking unsafe qualification session for one native Qwen3.5 main-model token.

use hipcore::{Device, DeviceBuffer};

use super::ResourceOwner;
use super::model_plan::{DeviceModelPlan, ModelDeviceByteDemand};
use super::model_resources::ModelDeviceResources;
use super::session::{Qwen35NativeSessionState, begin_error, completion_error, session_state};
use crate::{Qwen35Weights, Result};

/// Checked device-allocation demand for one native main-model qualification session.
///
/// Categories describe requested allocation extents for the one owned resource
/// bundle. They are neither observed residency nor a capacity, performance, or
/// device-qualification result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35NativeExecutionDeviceDemand {
    weights: usize,
    full_workspace: usize,
    recurrent_workspace: usize,
    finish_workspace: usize,
    hidden_rows: usize,
    final_normalized: usize,
    logits: usize,
    mrope_controls: usize,
    key_values: usize,
    page_table: usize,
    recurrent_history_active: usize,
    recurrent_history_staged: usize,
    recurrent_state_active: usize,
    recurrent_state_staged: usize,
    total: usize,
}

impl Qwen35NativeExecutionDeviceDemand {
    fn from_bytes(bytes: ModelDeviceByteDemand) -> Result<Self> {
        Ok(Self {
            weights: bytes.weights,
            full_workspace: bytes.full_workspace,
            recurrent_workspace: bytes.recurrent_workspace,
            finish_workspace: bytes.finish_workspace,
            hidden_rows: bytes.hidden_rows,
            final_normalized: bytes.final_normalized,
            logits: bytes.logits,
            mrope_controls: bytes.mrope_controls,
            key_values: bytes.key_values,
            page_table: bytes.page_table,
            recurrent_history_active: bytes.recurrent_history_active,
            recurrent_history_staged: bytes.recurrent_history_staged,
            recurrent_state_active: bytes.recurrent_state_active,
            recurrent_state_staged: bytes.recurrent_state_staged,
            total: bytes.total()?,
        })
    }

    /// Requested immutable embedding, layer, final-normalization, and output weight bytes.
    #[must_use]
    pub const fn weight_bytes(self) -> usize {
        self.weights
    }

    /// Requested reusable full-attention workspace bytes.
    #[must_use]
    pub const fn full_workspace_bytes(self) -> usize {
        self.full_workspace
    }

    /// Requested reusable recurrent-block workspace bytes.
    #[must_use]
    pub const fn recurrent_workspace_bytes(self) -> usize {
        self.recurrent_workspace
    }

    /// Requested reusable shared layer-finish workspace bytes.
    #[must_use]
    pub const fn finish_workspace_bytes(self) -> usize {
        self.finish_workspace
    }

    /// Requested hidden ping-pong-row bytes.
    #[must_use]
    pub const fn hidden_row_bytes(self) -> usize {
        self.hidden_rows
    }

    /// Requested terminal normalized-hidden bytes.
    #[must_use]
    pub const fn final_normalized_bytes(self) -> usize {
        self.final_normalized
    }

    /// Requested output-logit bytes.
    #[must_use]
    pub const fn logits_bytes(self) -> usize {
        self.logits
    }

    /// Requested shared text-mRoPE control-buffer bytes.
    #[must_use]
    pub const fn mrope_control_bytes(self) -> usize {
        self.mrope_controls
    }

    /// Requested native paged K/V backing bytes.
    #[must_use]
    pub const fn key_value_bytes(self) -> usize {
        self.key_values
    }

    /// Requested native paged-K/V table bytes.
    #[must_use]
    pub const fn page_table_bytes(self) -> usize {
        self.page_table
    }

    /// Requested active recurrent convolution-history bytes.
    #[must_use]
    pub const fn recurrent_history_active_bytes(self) -> usize {
        self.recurrent_history_active
    }

    /// Requested staged recurrent convolution-history bytes.
    #[must_use]
    pub const fn recurrent_history_staged_bytes(self) -> usize {
        self.recurrent_history_staged
    }

    /// Requested active recurrent-state bytes.
    #[must_use]
    pub const fn recurrent_state_active_bytes(self) -> usize {
        self.recurrent_state_active
    }

    /// Requested staged recurrent-state bytes.
    #[must_use]
    pub const fn recurrent_state_staged_bytes(self) -> usize {
        self.recurrent_state_staged
    }

    /// Checked total requested device bytes across this one owned session.
    #[must_use]
    pub const fn total_bytes(self) -> usize {
        self.total
    }
}

/// Artifact-bound native plan for one Qwen3.5 main-model baseline.
///
/// This plan borrows the exact verified weights that it will upload. Its block
/// sequence contains the artifact's main blocks and terminal normalization and
/// output head; optional auxiliary NextN blocks are intentionally outside that
/// baseline and are not a reason to reject an otherwise supported artifact.
#[derive(Debug)]
pub struct Qwen35NativeExecutionPlan<'weights, 'artifact> {
    weights: &'weights Qwen35Weights<'artifact>,
    plan: DeviceModelPlan,
    demand: Qwen35NativeExecutionDeviceDemand,
}

impl<'weights, 'artifact> Qwen35NativeExecutionPlan<'weights, 'artifact> {
    /// Derive a checked native main-model plan from verified Qwen3.5 weights.
    ///
    /// This performs no device initialization, allocation, upload, or kernel
    /// submission.
    ///
    /// # Errors
    ///
    /// Refuses unsupported main-block metadata, matrix formats or geometry,
    /// invalid context or page geometry, and unrepresentable allocation extents.
    pub fn try_from_weights(
        weights: &'weights Qwen35Weights<'artifact>,
        max_context: usize,
        page_tokens: kernels::attention::NativePageTokens,
    ) -> Result<Self> {
        let plan = DeviceModelPlan::from_weights(weights, max_context, page_tokens)?;
        let demand = Qwen35NativeExecutionDeviceDemand::from_bytes(plan.bytes)?;
        Ok(Self {
            weights,
            plan,
            demand,
        })
    }

    /// Return this plan's exact checked allocation request.
    #[must_use]
    pub const fn device_demand(&self) -> Qwen35NativeExecutionDeviceDemand {
        self.demand
    }

    /// Upload this plan's exact verified weights and create its owned session.
    ///
    /// # Errors
    ///
    /// Returns typed stream, allocation, upload, or verified weight-binding
    /// failures. This performs no inference submission.
    ///
    /// # Safety
    ///
    /// `device` must be the qualified `gfx1100` device selected for the native
    /// kernels. Construction establishes ownership only: it is not a physical
    /// GPU qualification, capacity grant, performance claim, or serving API.
    pub unsafe fn into_session(self, device: &Device) -> Result<Qwen35NativeExecutionSession> {
        let resources = ModelDeviceResources::new(self.weights, self.plan, device)?;
        Ok(Qwen35NativeExecutionSession {
            owner: ResourceOwner::new(resources),
        })
    }
}

/// One owned blocking native main-model qualification session.
///
/// The session owns the verified uploads, stream, full/recurrent workspaces,
/// K/V pool and table, recurrent active and staged state, and each pending
/// token's buffers. Submitted resources remain owned until completion is
/// synchronized and publication succeeds; uncertain completion retains or
/// forgets the complete bundle rather than releasing buffers early.
pub struct Qwen35NativeExecutionSession {
    owner: ResourceOwner<ModelDeviceResources>,
}

impl Qwen35NativeExecutionSession {
    /// Return the externally observable completion state.
    #[must_use]
    pub fn state(&self) -> Qwen35NativeSessionState {
        session_state(&self.owner)
    }

    /// Execute one token through all native main blocks and return its logits.
    ///
    /// The blocking step covers embedding, every admitted full or recurrent
    /// main block, final normalization, and the distinct output head. It does
    /// not execute optional NextN auxiliary blocks.
    ///
    /// # Errors
    ///
    /// Preflight refusal leaves a ready session unchanged. Any failure after
    /// the first submission permanently poisons the session; its state reports
    /// whether completion is known idle or remains uncertain. Failed work never
    /// publishes staged K/V or recurrent state and returns no partial logits.
    ///
    /// # Safety
    ///
    /// The caller guarantees a qualified `gfx1100` device and finite
    /// normal-or-zero values for every operand and intermediate in the native
    /// numerical domain, including verified weights, embedding rows, full and
    /// recurrent state, K/V, controls, workspace, normalization, and output.
    /// This one-token blocking qualification boundary does not independently
    /// establish numerical validity, hardware parity, performance, physical-GPU
    /// qualification, capacity, or serving safety.
    pub unsafe fn step(&mut self, token: u32) -> Result<DeviceBuffer<f32>> {
        let mut in_flight = self.owner.begin().map_err(begin_error)?;
        {
            let resources = in_flight.resource().map_err(completion_error)?;
            resources.prepare_step(token)?;
        }
        in_flight.mark_submitted();
        {
            let resources = in_flight.resource().map_err(completion_error)?;
            // SAFETY: the caller supplies the complete native numerical-domain and device qualification guarantees.
            unsafe { resources.submit_step()? };
        }
        in_flight
            .complete(publish_completed_step)
            .map_err(completion_error)
    }
}

fn publish_completed_step(resources: &mut ModelDeviceResources) -> Result<DeviceBuffer<f32>> {
    resources.publish_completed()
}
