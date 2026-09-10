//! Owned native resources for one whole Qwen3.5 main-model step.

use cache::{NativePagedKvBuffers, NativePagedKvPlan, NativePagedKvPool};
use core::mem::ManuallyDrop;
use hipcore::{
    BufferAllocationError, Device, DeviceBuffer, InventoryRelease, Stream, StreamCreationError,
    TeardownBuffer, TeardownInventory,
};
use snafu::{IntoError, ResultExt};
use std::sync::Arc;

use super::CompletionResource;
use super::custody::{
    NativeBufferParts, NativeBufferSink, NativeBuildGuard, NativeBuildResult, NativeBuildScope,
    NativeBuildSource,
};
use super::dispatch::{DeferredFullAttention, launch_rms_norm};
use super::finish::{LayerFinishWeights, LayerFinishWorkspace};
use super::model_plan::{DeviceModelPlan, NativeBlockPlan};
use super::model_session::NativeBuildFailure;
use super::model_step::ModelTokenPlan;
use super::recurrent::{
    DeferredRecurrent, NativeRecurrentState, NativeRecurrentWeights, NativeRecurrentWorkspace,
};
use super::resources::{FullAttentionStep, NativeWorkspace, native_mrope_controls};
use super::weights::{NativeMatrix, NativeWeights, f32_parameter_buffer};
use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationSnafu, NativeDeviceSnafu, NativeKernelSnafu,
    NativePagedKvSnafu, NativeSessionStateSnafu,
};
use crate::{Qwen35Weights, Result};

struct ModelStep {
    token: ModelTokenPlan,
    logits: DeviceBuffer<f32>,
    cosine: Option<DeviceBuffer<f32>>,
    sine: Option<DeviceBuffer<f32>>,
}

enum NativeModelLayer {
    Full(Box<NativeWeights>),
    Recurrent(Box<NativeRecurrentLayerWeights>),
}

struct NativeRecurrentLayerWeights {
    weights: NativeRecurrentWeights,
    finish: LayerFinishWeights,
}

enum NativeSessionLayer {
    Full,
    Recurrent(NativeRecurrentState),
}

/// Immutable native uploads retained by every session created from one model.
pub(super) struct NativeResidentModelResources {
    verified_weights: Qwen35Weights,
    plan: DeviceModelPlan,
    device: Device,
    embedding: NativeMatrix,
    output: NativeMatrix,
    output_norm: DeviceBuffer<f32>,
    layers: Vec<NativeModelLayer>,
}

/// One non-cloneable mutable native session retaining submitted work through completion.
pub(super) struct ModelSessionResources {
    model: Arc<NativeResidentModelResources>,
    plan: DeviceModelPlan,
    kv: Option<NativePagedKvPool>,
    stream: Stream,
    numerical_status: kernels::numerical_status::NativeNumericalStatus,
    full_workspace: Option<NativeWorkspace>,
    recurrent_workspace: Option<NativeRecurrentWorkspace>,
    finish_workspace: LayerFinishWorkspace,
    hidden_a: DeviceBuffer<f32>,
    hidden_b: DeviceBuffer<f32>,
    final_normalized: DeviceBuffer<f32>,
    layers: Vec<NativeSessionLayer>,
    step: Option<ModelStep>,
    failed_step: Option<NativeBufferParts>,
    failed_step_creation: Option<BufferAllocationError>,
    position: usize,
}

struct ResidentBuildFields {
    embedding: NativeMatrix,
    output: NativeMatrix,
    output_norm: DeviceBuffer<f32>,
    layers: Vec<NativeModelLayer>,
}

struct SessionBuildFields {
    numerical_status: kernels::numerical_status::NativeNumericalStatus,
    kv: Option<NativePagedKvPool>,
    full_workspace: Option<NativeWorkspace>,
    recurrent_workspace: Option<NativeRecurrentWorkspace>,
    finish_workspace: LayerFinishWorkspace,
    hidden_a: DeviceBuffer<f32>,
    hidden_b: DeviceBuffer<f32>,
    final_normalized: DeviceBuffer<f32>,
    layers: Vec<NativeSessionLayer>,
}

struct SessionBuildPrefix {
    numerical_status: NativeBuildGuard<kernels::numerical_status::NativeNumericalStatus>,
    kv: Option<NativeBuildGuard<NativePagedKvPool>>,
    full_workspace: Option<NativeBuildGuard<NativeWorkspace>>,
    recurrent_workspace: Option<NativeBuildGuard<NativeRecurrentWorkspace>>,
}

/// Shared immutable uploads retained by session teardown without an ordinary
/// native destructor reachable on abandonment.
///
/// A pending, synchronization-unconfirmed, or quarantined session stream may
/// still reference resident weights. Dropping public teardown custody must
/// therefore retain, rather than decrement the last resident `Arc` into the
/// ordinary `DeviceBuffer` drop path.
pub(super) struct ResidentRetention<T> {
    resident: ManuallyDrop<Arc<T>>,
}

impl<T> ResidentRetention<T> {
    pub(super) const fn new(resident: Arc<T>) -> Self {
        Self {
            resident: ManuallyDrop::new(resident),
        }
    }

    pub(super) fn get(&self) -> &Arc<T> {
        &self.resident
    }

    pub(super) fn recover(self) -> Arc<T> {
        ManuallyDrop::into_inner(self.resident)
    }
}

/// A successfully created stream disarmed during fallible construction.
pub(super) struct StreamRetention {
    stream: ManuallyDrop<Stream>,
}

impl StreamRetention {
    pub(super) const fn new(stream: Stream) -> Self {
        Self {
            stream: ManuallyDrop::new(stream),
        }
    }

    pub(super) fn recover(self) -> Stream {
        ManuallyDrop::into_inner(self.stream)
    }
}

/// Fully disarmed immutable uploads awaiting a dedicated teardown stream.
#[must_use = "resident uploads remain live until explicit HIP teardown"]
pub(super) struct NativeResidentTeardownParts {
    device: Device,
    buffers: Vec<TeardownBuffer>,
}

/// Explicit ownership for resident uploads after their last model `Arc` ended.
///
/// Like session teardown, this owns the HIP primitive directly and adds no
/// release semantics. Its buffer extent comes only from captured owners.
#[must_use = "resident teardown retains live native ownership"]
pub(super) enum NativeResidentTeardown {
    /// Creating a fresh owned teardown stream failed before HIP admission.
    Unadmitted { _parts: NativeResidentTeardownParts },
    /// Stream creation failed without establishing that no handle was returned.
    ///
    /// The complete creation error never enters ordinary teardown or the
    /// normal inventory. Retain unknown future error variants as conservatively
    /// as today's non-null-on-error quarantine, with the disarmed buffers.
    CreationQuarantined {
        _parts: NativeResidentTeardownParts,
        _error: StreamCreationError,
    },
    /// Checked accounting admitted a prefix and retained the remaining buffers.
    PartiallyAdmitted {
        _inventory: TeardownInventory,
        _buffers: Vec<TeardownBuffer>,
    },
    /// HIP's actual aggregate outcome for every admitted resident owner.
    Releasing(InventoryRelease),
}

/// Lossless session custody before or during explicit HIP aggregate teardown.
///
/// Every mutable buffer is first converted to an inert `TeardownBuffer`. The
/// resident `Arc` remains here until the stream's aggregate teardown reaches a
/// terminal outcome, so a last external model handle cannot free immutable
/// uploads while this session's stream may still reference them.
#[must_use = "native session teardown retains live native ownership"]
pub(super) enum ModelSessionTeardown {
    /// The supplied stream was not an owned HIP stream; no buffer was admitted.
    Unadmitted { _parts: ModelSessionTeardownParts },
    /// Some buffers were admitted and a checked accounting refusal retained the rest.
    ///
    /// This is deliberately opaque to the caller: both the admitted inventory
    /// and every rejected or unvisited inert owner remain retained, with no
    /// ordinary destructor reachable.
    PartiallyAdmitted {
        _inventory: TeardownInventory,
        _buffers: Vec<TeardownBuffer>,
        _resident: Option<ResidentRetention<NativeResidentModelResources>>,
        _creation_failure: Option<BufferAllocationError>,
    },
    /// The HIP inventory owns the stream and all session buffers; its exact
    /// release outcome is retained with the resident owner.
    Releasing {
        release: InventoryRelease,
        resident: Option<ResidentRetention<NativeResidentModelResources>>,
        creation_failure: Option<BufferAllocationError>,
    },
}

/// Observable aggregate-teardown custody status for one native model session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ModelSessionTeardownState {
    /// All session buffer and stream destructors were acknowledged.
    Released,
    /// Admission refused before HIP teardown accepted the stream.
    Unadmitted,
    /// Checked inventory accounting retained admitted and unadmitted ownership.
    PartiallyAdmitted,
    /// HIP preflight retained the aggregate without a destructor call.
    Pending,
    /// Stream completion is unproved and only reconciliation remains sound.
    SynchronizationUnconfirmed,
    /// A destructor outcome is indeterminate and remains conservatively charged.
    Quarantined,
}

impl ModelSessionTeardown {
    /// Classify the exact ownership retained by this teardown result.
    pub(super) const fn state(&self) -> ModelSessionTeardownState {
        if self.has_creation_quarantine() {
            return ModelSessionTeardownState::Quarantined;
        }
        self.known_state()
    }

    /// Classify only the independently retained stream and completed buffers.
    pub(super) const fn known_state(&self) -> ModelSessionTeardownState {
        match self {
            Self::Unadmitted { .. } => ModelSessionTeardownState::Unadmitted,
            Self::PartiallyAdmitted { .. } => ModelSessionTeardownState::PartiallyAdmitted,
            Self::Releasing { release, .. } => match release {
                InventoryRelease::Released(_) => ModelSessionTeardownState::Released,
                InventoryRelease::Pending(_) => ModelSessionTeardownState::Pending,
                InventoryRelease::SynchronizationUnconfirmed(_) => {
                    ModelSessionTeardownState::SynchronizationUnconfirmed
                }
                _ => ModelSessionTeardownState::Quarantined,
            },
        }
    }

    const fn has_creation_quarantine(&self) -> bool {
        match self {
            Self::Unadmitted { _parts: parts } => parts.creation_failure.is_some(),
            Self::PartiallyAdmitted {
                _creation_failure: creation_failure,
                ..
            }
            | Self::Releasing {
                creation_failure, ..
            } => creation_failure.is_some(),
        }
    }

    /// Forward HIP's only retryable aggregate transition without changing custody.
    pub(super) fn retry_pending(self) -> Self {
        match self {
            Self::Releasing {
                release: InventoryRelease::Pending(pending),
                resident,
                creation_failure,
            } => Self::Releasing {
                release: pending.retry(),
                resident,
                creation_failure,
            },
            other => other,
        }
    }

    /// Forward HIP's deliberate non-destructive synchronization reconciliation.
    pub(super) fn reconcile_synchronization(self) -> Self {
        match self {
            Self::Releasing {
                release: InventoryRelease::SynchronizationUnconfirmed(unconfirmed),
                resident,
                creation_failure,
            } => Self::Releasing {
                release: unconfirmed.reconcile(),
                resident,
                creation_failure,
            },
            other => other,
        }
    }

    /// Recover the exact resident owner only after HIP acknowledged the whole
    /// mutable-session inventory and its ordered stream.
    ///
    /// # Errors
    ///
    /// Returns the unchanged boxed teardown custody unless the inventory has
    /// an actual full HIP acknowledgement.
    pub(super) fn into_released_resident(
        self,
    ) -> core::result::Result<Arc<NativeResidentModelResources>, Box<Self>> {
        match self {
            Self::Releasing {
                release: InventoryRelease::Released(_),
                resident: Some(resident),
                creation_failure: None,
            } => Ok(resident.recover()),
            other => Err(Box::new(other)),
        }
    }

    /// Consume a fully acknowledged inventory and recover any resident anchor.
    pub(super) fn into_released(
        self,
    ) -> core::result::Result<Option<Arc<NativeResidentModelResources>>, Box<Self>> {
        match self {
            Self::Releasing {
                release: InventoryRelease::Released(_),
                resident,
                creation_failure: None,
            } => Ok(resident.map(ResidentRetention::recover)),
            other => Err(Box::new(other)),
        }
    }
}

impl NativeResidentTeardown {
    /// Classify the exact ownership retained by this resident teardown result.
    pub(super) const fn state(&self) -> ModelSessionTeardownState {
        match self {
            Self::Unadmitted { .. } => ModelSessionTeardownState::Unadmitted,
            Self::CreationQuarantined { .. } => ModelSessionTeardownState::Quarantined,
            Self::PartiallyAdmitted { .. } => ModelSessionTeardownState::PartiallyAdmitted,
            Self::Releasing(release) => match release {
                InventoryRelease::Released(_) => ModelSessionTeardownState::Released,
                InventoryRelease::Pending(_) => ModelSessionTeardownState::Pending,
                InventoryRelease::SynchronizationUnconfirmed(_) => {
                    ModelSessionTeardownState::SynchronizationUnconfirmed
                }
                _ => ModelSessionTeardownState::Quarantined,
            },
        }
    }

    /// Forward HIP's only retryable aggregate transition without changing custody.
    pub(super) fn retry_pending(self) -> Self {
        match self {
            Self::Releasing(InventoryRelease::Pending(pending)) => Self::Releasing(pending.retry()),
            other => other,
        }
    }

    /// Forward HIP's deliberate non-destructive synchronization reconciliation.
    pub(super) fn reconcile_synchronization(self) -> Self {
        match self {
            Self::Releasing(InventoryRelease::SynchronizationUnconfirmed(unconfirmed)) => {
                Self::Releasing(unconfirmed.reconcile())
            }
            other => other,
        }
    }

    /// Consume this custody only after every resident destructor was acknowledged.
    pub(super) fn into_released(self) -> core::result::Result<(), Box<Self>> {
        match self {
            Self::Releasing(InventoryRelease::Released(_)) => Ok(()),
            other => Err(Box::new(other)),
        }
    }
}

/// Fully disarmed mutable session buffers awaiting aggregate admission.
#[must_use = "all typed buffers have been disarmed and must enter explicit teardown"]
pub(super) struct ModelSessionTeardownParts {
    stream: Stream,
    buffers: Vec<TeardownBuffer>,
    resident: Option<ResidentRetention<NativeResidentModelResources>>,
    creation_failure: Option<BufferAllocationError>,
}

impl ModelSessionTeardownParts {
    pub(super) fn new(
        stream: Stream,
        buffers: Vec<TeardownBuffer>,
        resident: Option<ResidentRetention<NativeResidentModelResources>>,
        creation_failure: Option<BufferAllocationError>,
    ) -> Self {
        Self {
            stream,
            buffers,
            resident,
            creation_failure,
        }
    }

    /// Admit the owned stream before attempting any fallible buffer accounting.
    pub(super) fn begin_release(self) -> ModelSessionTeardown {
        let Self {
            stream,
            mut buffers,
            resident,
            creation_failure,
        } = self;
        let mut inventory = match TeardownInventory::try_new(stream) {
            Ok(inventory) => inventory,
            Err(stream) => {
                return ModelSessionTeardown::Unadmitted {
                    _parts: Self {
                        stream: stream.into_stream(),
                        buffers,
                        resident,
                        creation_failure,
                    },
                };
            }
        };
        while let Some(buffer) = buffers.pop() {
            if let Err(error) = inventory.push_teardown_buffer(buffer) {
                // `pop` left a spare slot, so preserving the rejected owner
                // cannot allocate or drop it through an ordinary destructor.
                buffers.push(error.into_buffer());
                return ModelSessionTeardown::PartiallyAdmitted {
                    _inventory: inventory,
                    _buffers: buffers,
                    _resident: resident,
                    _creation_failure: creation_failure,
                };
            }
        }
        ModelSessionTeardown::Releasing {
            release: inventory.begin_release(),
            resident,
            creation_failure,
        }
    }
}

impl NativeResidentTeardownParts {
    pub(super) fn new(device: Device, buffers: Vec<TeardownBuffer>) -> Self {
        Self { device, buffers }
    }

    /// Create the owned empty stream needed to quiesce an otherwise unique model.
    pub(super) fn begin_release(self) -> NativeResidentTeardown {
        let Self {
            device,
            mut buffers,
        } = self;
        let stream = match Stream::new_tracked(&device) {
            Ok(stream) => stream,
            Err(StreamCreationError::NoHandle(_)) => {
                return NativeResidentTeardown::Unadmitted {
                    _parts: Self { device, buffers },
                };
            }
            Err(error) => {
                return NativeResidentTeardown::CreationQuarantined {
                    _parts: Self { device, buffers },
                    _error: error,
                };
            }
        };
        let mut inventory = match TeardownInventory::try_new(stream) {
            Ok(inventory) => inventory,
            Err(stream) => {
                drop(stream);
                return NativeResidentTeardown::Unadmitted {
                    _parts: Self { device, buffers },
                };
            }
        };
        while let Some(buffer) = buffers.pop() {
            if let Err(error) = inventory.push_teardown_buffer(buffer) {
                buffers.push(error.into_buffer());
                return NativeResidentTeardown::PartiallyAdmitted {
                    _inventory: inventory,
                    _buffers: buffers,
                };
            }
        }
        NativeResidentTeardown::Releasing(inventory.begin_release())
    }
}

fn build_resident_fields(
    weights: &Qwen35Weights,
    plan: &DeviceModelPlan,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<ResidentBuildFields> {
    plan.bytes.total().map_err(NativeBuildSource::decoder)?;
    let embedding = scope.guard(
        NativeMatrix::upload(weights, &plan.embedding, device, scope)?,
        NativeMatrix::into_buffer_sink,
    );
    let output = scope.guard(
        NativeMatrix::upload(weights, &plan.output, device, scope)?,
        NativeMatrix::into_buffer_sink,
    );
    let output_norm = scope.guard(
        f32_parameter_buffer(weights, &plan.output_norm, device, scope)?,
        |buffer, sink| sink.push_f32(buffer),
    );
    let layers = upload_resident_layers(weights, plan, device, scope)?;
    Ok(ResidentBuildFields {
        embedding: embedding.commit(),
        output: output.commit(),
        output_norm: output_norm.commit(),
        layers,
    })
}

fn build_session_fields(
    model: &Arc<NativeResidentModelResources>,
    plan: &DeviceModelPlan,
    scope: &NativeBuildScope,
) -> NativeBuildResult<SessionBuildFields> {
    let numerical_status = scope.guard(
        build_numerical_status(&model.device, scope)?,
        retain_numerical_status,
    );
    let kv = plan
        .kv
        .map(|kv_plan| build_native_kv(kv_plan, &model.device, scope))
        .transpose()?
        .map(|pool| scope.guard(pool, retain_native_kv));
    let full_workspace = plan
        .full_workspace
        .as_ref()
        .map(|workspace| NativeWorkspace::new(workspace, &model.device, scope))
        .transpose()?
        .map(|workspace| scope.guard(workspace, NativeWorkspace::into_buffer_sink));
    let recurrent_workspace = plan
        .recurrent_workspace
        .as_ref()
        .map(|workspace| NativeRecurrentWorkspace::new(workspace, &model.device, scope))
        .transpose()?
        .map(|workspace| scope.guard(workspace, NativeRecurrentWorkspace::into_buffer_sink));
    let prefix = SessionBuildPrefix {
        numerical_status,
        kv,
        full_workspace,
        recurrent_workspace,
    };
    build_remaining_session_fields(model, plan, scope, prefix)
}

fn build_remaining_session_fields(
    model: &Arc<NativeResidentModelResources>,
    plan: &DeviceModelPlan,
    scope: &NativeBuildScope,
    prefix: SessionBuildPrefix,
) -> NativeBuildResult<SessionBuildFields> {
    let finish_workspace = scope.guard(
        LayerFinishWorkspace::new(plan.finish_workspace, &model.device, scope)?,
        LayerFinishWorkspace::into_buffer_sink,
    );
    let hidden_a = scope.allocate_f32(&model.device, plan.layout.hidden)?;
    let hidden_b = scope.allocate_f32(&model.device, plan.layout.hidden)?;
    let final_normalized = scope.allocate_f32(&model.device, plan.output_rms.elements())?;
    let layers = allocate_session_layers(plan, &model.device, scope)?;
    Ok(SessionBuildFields {
        numerical_status: prefix.numerical_status.commit(),
        kv: prefix.kv.map(NativeBuildGuard::commit),
        full_workspace: prefix.full_workspace.map(NativeBuildGuard::commit),
        recurrent_workspace: prefix.recurrent_workspace.map(NativeBuildGuard::commit),
        finish_workspace: finish_workspace.commit(),
        hidden_a: hidden_a.commit(),
        hidden_b: hidden_b.commit(),
        final_normalized: final_normalized.commit(),
        layers,
    })
}

pub(super) fn build_numerical_status(
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<kernels::numerical_status::NativeNumericalStatus> {
    let bits = scope.allocate_u32(device, 1)?;
    let uninitialized = match kernels::numerical_status::NativeNumericalStatus::try_bind_buffer(
        device,
        bits.commit(),
    ) {
        Ok(status) => status,
        Err(error) => {
            let kind = error.kind();
            scope.retain(error.into_buffer(), |buffer, sink| sink.push_u32(buffer));
            let rule = match kind {
                kernels::numerical_status::NativeNumericalStatusBufferErrorKind::DeviceMismatch => {
                    "native status construction buffer must belong to its session device"
                }
                kernels::numerical_status::NativeNumericalStatusBufferErrorKind::LengthMismatch => {
                    "native status construction buffer must contain exactly one word"
                }
            };
            return Err(NativeBuildSource::decoder(
                NativeSessionStateSnafu { rule }.build(),
            ));
        }
    };
    match uninitialized.initialize() {
        Ok(status) => Ok(status),
        Err(error) => {
            let (source, bits) = error.into_parts();
            scope.retain(bits, |buffer, sink| sink.push_u32(buffer));
            Err(NativeBuildSource::decoder(
                NativeDeviceSnafu.into_error(source),
            ))
        }
    }
}

pub(super) fn build_native_kv(
    plan: NativePagedKvPlan,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<NativePagedKvPool> {
    let keys = scope.allocate_f32(device, plan.key_value_elements())?;
    let values = scope.allocate_f32(device, plan.key_value_elements())?;
    let table = scope.allocate_u32(device, plan.page_table_elements())?;
    let buffers = NativePagedKvBuffers::new(keys.commit(), values.commit(), table.commit());
    match NativePagedKvPool::try_from_buffers(plan, buffers) {
        Ok(pool) => Ok(pool),
        Err(error) => {
            let (source, buffers) = error.into_parts();
            scope.retain(buffers, retain_native_kv_buffers);
            Err(NativeBuildSource::decoder(
                NativePagedKvSnafu.into_error(source),
            ))
        }
    }
}

fn retain_native_kv(pool: NativePagedKvPool, sink: &mut NativeBufferParts) {
    retain_native_kv_buffers(pool.into_buffers(), sink);
}

fn retain_native_kv_buffers(buffers: NativePagedKvBuffers, sink: &mut NativeBufferParts) {
    let (keys, values, table) = buffers.into_parts();
    sink.push_f32(keys);
    sink.push_f32(values);
    sink.push_u32(table);
}

fn retain_numerical_status(
    status: kernels::numerical_status::NativeNumericalStatus,
    sink: &mut NativeBufferParts,
) {
    sink.push_u32(status.into_buffer());
}

impl NativeResidentModelResources {
    pub(super) fn new(
        weights: &Qwen35Weights,
        plan: DeviceModelPlan,
        device: &Device,
    ) -> core::result::Result<Self, NativeBuildFailure> {
        let scope = NativeBuildScope::new();
        match Self::build(weights, plan, device, &scope) {
            Ok(resources) => Ok(resources),
            Err(source) => Err(NativeBuildFailure::resident(source, scope, device.clone())),
        }
    }

    fn build(
        weights: &Qwen35Weights,
        plan: DeviceModelPlan,
        device: &Device,
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        let fields = build_resident_fields(weights, &plan, device, scope)?;
        let ResidentBuildFields {
            embedding,
            output,
            output_norm,
            layers,
        } = fields;
        Ok(Self {
            verified_weights: weights.clone(),
            plan,
            device: device.clone(),
            embedding,
            output,
            output_norm,
            layers,
        })
    }

    pub(super) fn plan_session(&self, max_context: usize) -> Result<DeviceModelPlan> {
        derive_session_plan(&self.verified_weights, &self.plan, max_context)
    }

    pub(super) const fn context_ceiling(&self) -> usize {
        self.plan.layout.max_context()
    }

    /// Disarm the final unique resident owner before it creates teardown work.
    ///
    /// The caller establishes uniqueness by consuming the last `Arc`; no
    /// session can then retain a stream that references these uploads.
    pub(super) fn into_teardown_parts(self) -> NativeResidentTeardownParts {
        let Self {
            verified_weights,
            plan,
            device,
            embedding,
            output,
            output_norm,
            layers,
        } = self;
        drop(verified_weights);
        drop(plan);
        let mut buffers = NativeBufferParts::new();
        embedding.into_buffer_sink(&mut buffers);
        output.into_buffer_sink(&mut buffers);
        buffers.push_f32(output_norm);
        for layer in layers {
            layer.into_buffer_sink(&mut buffers);
        }
        NativeResidentTeardownParts {
            device,
            buffers: buffers.into_buffers(),
        }
    }
}

impl ModelSessionResources {
    pub(super) fn new(
        model: Arc<NativeResidentModelResources>,
        plan: DeviceModelPlan,
    ) -> core::result::Result<Self, NativeBuildFailure> {
        let scope = NativeBuildScope::new();
        let resident = ResidentRetention::new(model);
        let stream = match Stream::new_tracked(&resident.get().device) {
            Ok(stream) => StreamRetention::new(stream),
            Err(error) => {
                return Err(NativeBuildFailure::session(
                    NativeBuildSource::stream(error),
                    scope,
                    None,
                    resident,
                ));
            }
        };
        let fields = build_session_fields(resident.get(), &plan, &scope);
        let fields = match fields {
            Ok(fields) => fields,
            Err(source) => {
                return Err(NativeBuildFailure::session(
                    source,
                    scope,
                    Some(stream),
                    resident,
                ));
            }
        };
        let SessionBuildFields {
            numerical_status,
            kv,
            full_workspace,
            recurrent_workspace,
            finish_workspace,
            hidden_a,
            hidden_b,
            final_normalized,
            layers,
        } = fields;
        Ok(Self {
            model: resident.recover(),
            plan,
            kv,
            stream: stream.recover(),
            numerical_status,
            full_workspace,
            recurrent_workspace,
            finish_workspace,
            hidden_a,
            hidden_b,
            final_normalized,
            layers,
            step: None,
            failed_step: None,
            failed_step_creation: None,
            position: 0,
        })
    }

    /// Disarm every mutable allocation before attempting explicit teardown.
    ///
    /// No HIP operation occurs here. In particular, this path remains safe for
    /// a session with submitted work because every extracted buffer becomes an
    /// inert owner and the ordered stream is retained for later aggregate
    /// quiescence and release.
    pub(super) fn into_teardown_parts(self) -> ModelSessionTeardownParts {
        let Self {
            model,
            plan,
            kv,
            stream,
            numerical_status,
            full_workspace,
            recurrent_workspace,
            finish_workspace,
            hidden_a,
            hidden_b,
            final_normalized,
            layers,
            step,
            failed_step,
            failed_step_creation,
            position: _,
        } = self;
        drop(plan);
        let mut buffers = NativeBufferParts::new();
        if let Some(kv) = kv {
            let (keys, values, table) = kv.into_buffers().into_parts();
            buffers.push_f32(keys);
            buffers.push_f32(values);
            buffers.push_u32(table);
        }
        buffers.push_u32(numerical_status.into_buffer());
        if let Some(workspace) = full_workspace {
            workspace.into_buffer_sink(&mut buffers);
        }
        if let Some(workspace) = recurrent_workspace {
            workspace.into_buffer_sink(&mut buffers);
        }
        finish_workspace.into_buffer_sink(&mut buffers);
        buffers.push_f32(hidden_a);
        buffers.push_f32(hidden_b);
        buffers.push_f32(final_normalized);
        for layer in layers {
            layer.into_buffer_sink(&mut buffers);
        }
        if let Some(step) = step {
            step.into_buffer_sink(&mut buffers);
        }
        if let Some(failed_step) = failed_step {
            failed_step.append_to(&mut buffers);
        }
        ModelSessionTeardownParts {
            stream,
            buffers: buffers.into_buffers(),
            resident: Some(ResidentRetention::new(model)),
            creation_failure: failed_step_creation,
        }
    }

    pub(super) fn prepare_step(&mut self, token: u32) -> Result<()> {
        self.ensure_step_available()?;
        let token = ModelTokenPlan::from_model(&self.plan, self.position, token)?;
        let controls = self.attention_control_values(token)?;
        let mut failed_step = NativeBufferParts::new();
        let device = self.stream.device().clone();
        let (cosine, sine) = match controls {
            Some((cosine, sine)) => {
                let cosine = match copy_f32_to_device(
                    &device,
                    &cosine,
                    &mut failed_step,
                    &mut self.failed_step_creation,
                ) {
                    Ok(cosine) => cosine,
                    Err(error) => {
                        if !failed_step.is_empty() {
                            self.failed_step = Some(failed_step);
                        }
                        return Err(error);
                    }
                };
                let sine = match copy_f32_to_device(
                    &device,
                    &sine,
                    &mut failed_step,
                    &mut self.failed_step_creation,
                ) {
                    Ok(sine) => sine,
                    Err(error) => {
                        failed_step.push_f32(cosine);
                        self.failed_step = Some(failed_step);
                        return Err(error);
                    }
                };
                (Some(cosine), Some(sine))
            }
            None => (None, None),
        };
        let logits = match tracked_step_buffer(
            &device,
            self.model.output.shape.rows(),
            &mut self.failed_step_creation,
        ) {
            Ok(logits) => logits,
            Err(error) => {
                retain_optional_controls(cosine, sine, &mut failed_step);
                if !failed_step.is_empty() {
                    self.failed_step = Some(failed_step);
                }
                return Err(error);
            }
        };
        self.step = Some(ModelStep {
            token,
            logits,
            cosine,
            sine,
        });
        Ok(())
    }

    fn ensure_step_available(&self) -> Result<()> {
        if self.step.is_some() {
            return NativeSessionStateSnafu {
                rule: "native model step requires no pending output",
            }
            .fail();
        }
        if self.failed_step.is_some() {
            return NativeSessionStateSnafu {
                rule: "native session with failed step allocation requires explicit teardown",
            }
            .fail();
        }
        if self.failed_step_creation.is_some() {
            return NativeSessionStateSnafu {
                rule: "native session with quarantined step creation requires explicit teardown",
            }
            .fail();
        }
        Ok(())
    }

    /// True when a failed step allocation has already moved buffers into
    /// explicit inert custody and the session must not dispatch again.
    pub(super) const fn requires_teardown(&self) -> bool {
        self.failed_step.is_some() || self.failed_step_creation.is_some()
    }

    /// Submit one token through the complete artifact-ordered native main model.
    ///
    /// # Safety
    ///
    /// The complete owned bundle, including serialized weights, controls,
    /// staged K/V, recurrent state, input lookup output, and logits, remains
    /// exclusively owned on this ordered stream until completion is proved.
    /// Every native primitive receives this session's sticky checked-status
    /// allocation. Its post-sync read precedes all logical publication.
    #[expect(
        clippy::too_many_lines,
        reason = "one ordered native submission must retain the full model transaction and its single KV append"
    )]
    pub(super) unsafe fn submit_step(&mut self) -> Result<()> {
        let step = self.step.as_ref().ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native model submission requires a prepared step",
            }
            .build()
        })?;
        // SAFETY: this bundle owns the exact serialized embedding matrix,
        // first hidden row, and ordered stream through completion.
        unsafe {
            kernels::row_gemv::launch_row_decode_f32_checked(
                step.token.embedding,
                self.model.embedding.bytes.as_device_ptr(),
                self.model.embedding.bytes.len(),
                self.hidden_a.as_device_ptr(),
                self.hidden_a.len(),
                &self.stream,
                &self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)?;

        let mut append = self
            .kv
            .as_mut()
            .map(|pool| {
                // SAFETY: this model owns the pool and its exact ordered stream through completion.
                unsafe { pool.begin_append(1, &self.stream) }
            })
            .transpose()
            .context(NativePagedKvSnafu)?;
        let (mut input, mut output) = (&self.hidden_a, &self.hidden_b);
        let mut full_layer = 0_usize;
        for ((plan, layer), session_layer) in self
            .plan
            .layers
            .iter()
            .zip(&self.model.layers)
            .zip(&self.layers)
        {
            match (plan, layer, session_layer) {
                (
                    NativeBlockPlan::Full(plan),
                    NativeModelLayer::Full(weights),
                    NativeSessionLayer::Full,
                ) => {
                    let append = append.as_mut().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires one model KV append",
                        }
                        .build()
                    })?;
                    let workspace = self.full_workspace.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires model full workspace",
                        }
                        .build()
                    })?;
                    let cosine = step.cosine.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires prepared MRoPE cosine controls",
                        }
                        .build()
                    })?;
                    let sine = step.sine.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires prepared MRoPE sine controls",
                        }
                        .build()
                    })?;
                    let attention = step.token.attention.ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires prepared paged attention controls",
                        }
                        .build()
                    })?;
                    let deferred = DeferredFullAttention {
                        weights,
                        workspace,
                        finish_plan: &plan.finish,
                        finish_weights: &weights.finish,
                        finish_workspace: &self.finish_workspace,
                        plan: plan.workspace,
                        step: FullAttentionStep {
                            input,
                            output,
                            cosine,
                            sine,
                            attention,
                        },
                        stream: &self.stream,
                        numerical_status: &self.numerical_status,
                        full_layer,
                    };
                    // SAFETY: the enclosing submission owns every borrowed span through completion.
                    unsafe { deferred.submit(append) }?;
                    full_layer = full_layer.checked_add(1).ok_or_else(|| {
                        ArithmeticOverflowSnafu {
                            context: "native full-attention layer cursor",
                        }
                        .build()
                    })?;
                }
                (
                    NativeBlockPlan::Recurrent(recurrent),
                    NativeModelLayer::Recurrent(resources),
                    NativeSessionLayer::Recurrent(state),
                ) => {
                    let workspace = self.recurrent_workspace.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native recurrent block requires model recurrent workspace",
                        }
                        .build()
                    })?;
                    let deferred = DeferredRecurrent {
                        plan: &recurrent.plan,
                        weights: &resources.weights,
                        workspace,
                        state,
                        input,
                        output,
                        finish_plan: &recurrent.finish,
                        finish_weights: &resources.finish,
                        finish_workspace: &self.finish_workspace,
                        stream: &self.stream,
                        numerical_status: &self.numerical_status,
                    };
                    // SAFETY: the enclosing submission retains all staged recurrent state through completion.
                    unsafe { deferred.submit() }?;
                }
                _ => {
                    return NativeSessionStateSnafu {
                        rule: "native model resource roles must exactly bind planned block roles",
                    }
                    .fail();
                }
            }
            core::mem::swap(&mut input, &mut output);
        }
        // SAFETY: this model owns the exact final hidden, norm, and logits spans through completion.
        unsafe {
            launch_rms_norm(
                self.plan.output_rms,
                input,
                &self.model.output_norm,
                &self.final_normalized,
                &self.stream,
                &self.numerical_status,
            )
        }?;
        // SAFETY: the verified output matrix and exact final/logit spans remain owned through completion.
        unsafe {
            self.model.output.launch(
                &self.final_normalized,
                &step.logits,
                &self.stream,
                &self.numerical_status,
            )
        }?;
        if let Some(append) = append {
            append.prepare_commit().context(NativePagedKvSnafu)?;
        }
        Ok(())
    }

    pub(super) fn publish_completed(&mut self) -> Result<DeviceBuffer<f32>> {
        let step = self.step.take().ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native model publication requires one prepared step",
            }
            .build()
        })?;
        let next_position = step.token.next_position;
        if let Some(kv) = self.kv.as_mut() {
            // SAFETY: the resource owner synchronized this exact stream; every fallible local check and output extraction precedes host-ledger publication.
            unsafe {
                kv.commit_prepared_after_completion()
                    .context(NativePagedKvSnafu)?;
            }
        }
        for layer in &mut self.layers {
            if let NativeSessionLayer::Recurrent(state) = layer {
                state.publish_completed();
            }
        }
        self.position = next_position;
        Ok(step.logits)
    }

    fn attention_control_values(
        &self,
        token: ModelTokenPlan,
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        if token.attention.is_none() {
            return Ok(None);
        }
        let Some(workspace) = self.plan.full_workspace else {
            return NativeSessionStateSnafu {
                rule: "native model paged attention requires full-attention workspace",
            }
            .fail();
        };
        let (cosine, sine) = native_mrope_controls(
            self.plan.layout.text_mrope(),
            self.position,
            workspace.query_rotary.coefficient_elements(),
        )?;
        Ok(Some((cosine, sine)))
    }
}

/// Allocate then synchronously copy one host control vector without losing a
/// successful allocation when HIP reports a copy failure.
fn copy_f32_to_device(
    device: &Device,
    values: &[f32],
    failed_step: &mut NativeBufferParts,
    creation_failure: &mut Option<BufferAllocationError>,
) -> Result<DeviceBuffer<f32>> {
    let mut buffer = tracked_step_buffer(device, values.len(), creation_failure)?;
    match buffer.copy_from_host(values) {
        Ok(()) => Ok(buffer),
        Err(source) => {
            failed_step.push_f32(buffer);
            Err(source).context(NativeDeviceSnafu)
        }
    }
}

fn retain_optional_controls(
    cosine: Option<DeviceBuffer<f32>>,
    sine: Option<DeviceBuffer<f32>>,
    failed_step: &mut NativeBufferParts,
) {
    if let Some(cosine) = cosine {
        failed_step.push_f32(cosine);
    }
    if let Some(sine) = sine {
        failed_step.push_f32(sine);
    }
}

fn tracked_step_buffer<T: hipcore::BytePod>(
    device: &Device,
    elements: usize,
    creation_failure: &mut Option<BufferAllocationError>,
) -> Result<DeviceBuffer<T>> {
    match DeviceBuffer::alloc_tracked(device, elements) {
        Ok(buffer) => Ok(buffer),
        Err(BufferAllocationError::NoHandle(source)) => Err(source).context(NativeDeviceSnafu),
        Err(error) => {
            *creation_failure = Some(error);
            NativeSessionStateSnafu {
                rule: "native step allocation returned terminal creation quarantine",
            }
            .fail()
        }
    }
}

impl ModelStep {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_f32(self.logits);
        if let Some(cosine) = self.cosine {
            sink.push_f32(cosine);
        }
        if let Some(sine) = self.sine {
            sink.push_f32(sine);
        }
    }
}

impl NativeModelLayer {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        match self {
            Self::Full(weights) => weights.into_buffer_sink(sink),
            Self::Recurrent(weights) => weights.into_buffer_sink(sink),
        }
    }
}

impl NativeRecurrentLayerWeights {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        self.weights.into_buffer_sink(sink);
        self.finish.into_buffer_sink(sink);
    }
}

impl NativeSessionLayer {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        match self {
            Self::Full => {}
            Self::Recurrent(state) => state.into_buffer_sink(sink),
        }
    }
}

impl CompletionResource for ModelSessionResources {
    type Error = crate::Error;

    fn synchronize(&mut self) -> Result<()> {
        self.stream.synchronize().context(NativeDeviceSnafu)
    }

    fn validate_after_synchronization(&mut self) -> Result<()> {
        self.numerical_status
            .read_after_synchronization()
            .context(NativeKernelSnafu)
    }
}

fn derive_session_plan(
    weights: &Qwen35Weights,
    resident: &DeviceModelPlan,
    max_context: usize,
) -> Result<DeviceModelPlan> {
    if max_context > resident.layout.max_context() {
        return NativeSessionStateSnafu {
            rule: "native session context must not exceed its resident model ceiling",
        }
        .fail();
    }
    let session = DeviceModelPlan::from_weights(weights, max_context, resident.page_tokens)?;
    if session.embedding.shape != resident.embedding.shape
        || session.output.shape != resident.output.shape
        || session.output_rms.elements() != resident.output_rms.elements()
        || !same_layer_roles(&session.layers, &resident.layers)
    {
        return NativeSessionStateSnafu {
            rule: "native session plan must retain resident uploaded-weight bindings",
        }
        .fail();
    }
    Ok(session)
}

fn same_layer_roles(session: &[NativeBlockPlan], resident: &[NativeBlockPlan]) -> bool {
    session.len() == resident.len()
        && session.iter().zip(resident).all(|(session, resident)| {
            matches!(
                (session, resident),
                (NativeBlockPlan::Full(_), NativeBlockPlan::Full(_))
                    | (NativeBlockPlan::Recurrent(_), NativeBlockPlan::Recurrent(_))
            )
        })
}

fn upload_resident_layers(
    weights: &Qwen35Weights,
    plan: &DeviceModelPlan,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<Vec<NativeModelLayer>> {
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(plan.layers.len())
        .context(ExecutionAllocationSnafu {
            target: "native model layer resources",
            length: plan.layers.len(),
        })
        .map_err(NativeBuildSource::decoder)?;
    let mut layers = scope.guard(layers, retain_model_layers);
    for block in &plan.layers {
        let layer = match block {
            NativeBlockPlan::Full(plan) => NativeModelLayer::Full(Box::new(NativeWeights::upload(
                weights, plan, device, scope,
            )?)),
            NativeBlockPlan::Recurrent(block) => {
                let recurrent = scope.guard(
                    NativeRecurrentWeights::upload(weights, &block.plan, device, scope)?,
                    NativeRecurrentWeights::into_buffer_sink,
                );
                let finish = scope.guard(
                    LayerFinishWeights::upload(weights, &block.finish, device, scope)?,
                    LayerFinishWeights::into_buffer_sink,
                );
                NativeModelLayer::Recurrent(Box::new(NativeRecurrentLayerWeights {
                    weights: recurrent.commit(),
                    finish: finish.commit(),
                }))
            }
        };
        layers.push(layer);
    }
    if layers.len() != plan.layers.len() {
        let error = NativeSessionStateSnafu {
            rule: "native model resource layers must exactly cover every planned block",
        }
        .build();
        return Err(NativeBuildSource::decoder(error));
    }
    Ok(layers.commit())
}

fn allocate_session_layers(
    plan: &DeviceModelPlan,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<Vec<NativeSessionLayer>> {
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(plan.layers.len())
        .context(ExecutionAllocationSnafu {
            target: "native model session layers",
            length: plan.layers.len(),
        })
        .map_err(NativeBuildSource::decoder)?;
    let mut layers = scope.guard(layers, retain_session_layers);
    for block in &plan.layers {
        layers.push(match block {
            NativeBlockPlan::Full(_) => NativeSessionLayer::Full,
            NativeBlockPlan::Recurrent(recurrent) => NativeSessionLayer::Recurrent(
                NativeRecurrentState::new(&recurrent.plan, device, scope)?,
            ),
        });
    }
    if layers.len() != plan.layers.len() {
        let error = NativeSessionStateSnafu {
            rule: "native model session layers must exactly cover every planned block",
        }
        .build();
        return Err(NativeBuildSource::decoder(error));
    }
    Ok(layers.commit())
}

fn retain_model_layers(layers: Vec<NativeModelLayer>, sink: &mut NativeBufferParts) {
    for layer in layers {
        layer.into_buffer_sink(sink);
    }
}

fn retain_session_layers(layers: Vec<NativeSessionLayer>, sink: &mut NativeBufferParts) {
    for layer in layers {
        layer.into_buffer_sink(sink);
    }
}

#[cfg(test)]
mod tests {
    use super::derive_session_plan;
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{canonical_hybrid_fixture_with_context, verify_fixture};
    use crate::qwen35_native::model_plan::DeviceModelPlan;
    use crate::qwen35_native::model_step::ModelTokenPlan;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const RESIDENT_CONTEXT: usize = 16;
    const SESSION_CONTEXT: usize = 4;
    const PAGE_TOKENS: kernels::attention::NativePageTokens =
        kernels::attention::NativePageTokens::B8;

    #[test]
    fn session_plan_reuses_resident_binding_at_exact_smaller_context()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture_with_context(RESIDENT_CONTEXT)?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let resident = DeviceModelPlan::from_weights(&weights, RESIDENT_CONTEXT, PAGE_TOKENS)
            .map_err(|error| error.to_string())?;
        let session = derive_session_plan(&weights, &resident, SESSION_CONTEXT)
            .map_err(|error| error.to_string())?;

        assert_eq!(session.layout.max_context(), SESSION_CONTEXT);
        assert_eq!(session.page_tokens, resident.page_tokens);
        assert_eq!(session.layers.len(), resident.layers.len());
        assert!(
            session.bytes.key_values < resident.bytes.key_values,
            "a smaller exact session context must allocate its own smaller K/V extent"
        );
        assert!(
            ModelTokenPlan::from_model(&session, SESSION_CONTEXT, 0).is_err(),
            "the per-use plan cannot dispatch beyond its exact requested context"
        );
        assert!(
            derive_session_plan(&weights, &resident, RESIDENT_CONTEXT + 1).is_err(),
            "a session plan cannot exceed the resident model's immutable ceiling"
        );
        Ok(())
    }

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn abandoned_retention_never_drops_the_last_resident() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let resident = Arc::new(DropProbe(Arc::clone(&drops)));
            let _retention = super::ResidentRetention::new(Arc::clone(&resident));
            drop(resident);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn released_retention_recovers_the_exact_resident_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let resident = Arc::new(DropProbe(Arc::clone(&drops)));
        let retention = super::ResidentRetention::new(Arc::clone(&resident));
        drop(resident);
        let recovered = retention.recover();
        assert_eq!(Arc::strong_count(&recovered), 1);
        drop(recovered);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
