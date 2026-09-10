//! Shared native text execution with explicit per-use custody.

use std::fmt;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use decoders::{
    NativeBuildFailure, Qwen35NativeExecutionDeviceDemand, Qwen35NativeExecutionModel,
    Qwen35NativeExecutionModelClose, Qwen35NativeExecutionModelTeardown, Qwen35NativeExecutionPlan,
    Qwen35NativeExecutionSession, Qwen35NativeExecutionSessionPlan,
    Qwen35NativeExecutionSessionTeardown, Qwen35NativeExecutionSessionTeardownState,
};
use hipcore::{BufferRelease, Device, DeviceBuffer};
use kernels::attention::NativePageTokens;
use snafu::Snafu;
use text::{
    Cancellation, Generation, PreparedGeneration, RecycledGenerationDriver,
    RecycledGenerationError, RecycledLogitsPlan, RecycledLogitsStorage, TextPipeline,
};

/// Shared immutable native text execution owner.
///
/// This binds one exact [`TextPipeline`] profile to one immutable native model
/// upload. It is not a service, host-grant verifier, admission decision, or
/// residency claim.
pub struct NativeTextResident {
    inner: Arc<NativeTextResidentInner>,
}

struct NativeTextResidentInner {
    pipeline: TextPipeline,
    model: ExplicitReleaseOwner<Qwen35NativeExecutionModel>,
}

/// Keeps ordinary destruction unreachable until an explicit consuming path.
struct ExplicitReleaseOwner<T> {
    value: ManuallyDrop<T>,
}

impl<T> ExplicitReleaseOwner<T> {
    const fn new(value: T) -> Self {
        Self {
            value: ManuallyDrop::new(value),
        }
    }

    fn get(&self) -> &T {
        &self.value
    }

    fn into_inner(self) -> T {
        ManuallyDrop::into_inner(self.value)
    }
}

/// Failure while creating a shared native text resident.
#[must_use = "native construction failure retains its exact typed source"]
#[derive(Snafu)]
#[non_exhaustive]
pub enum NativeTextResidentBuildFailure {
    /// The exact pipeline profile could not form a native execution plan.
    #[snafu(display("native text plan failed: {source}"))]
    Plan {
        /// Original checked decoder-plan failure.
        source: Box<decoders::Error>,
        /// Exact text pipeline retained by the failed construction.
        pipeline: TextPipeline,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Native creation failed after retaining typed partial native custody.
    #[snafu(display("native text resident construction failed: {source}"))]
    Native {
        /// Original typed native construction custody.
        source: Box<NativeBuildFailure>,
        /// Exact text pipeline retained by the failed construction.
        pipeline: TextPipeline,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

impl fmt::Debug for NativeTextResidentBuildFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan { source, .. } => formatter
                .debug_struct("NativeTextResidentBuildFailure::Plan")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::Native { source, .. } => formatter
                .debug_struct("NativeTextResidentBuildFailure::Native")
                .field("source", source)
                .finish_non_exhaustive(),
        }
    }
}

/// Result of explicitly closing a native text resident.
#[must_use = "resident close retains the model or its explicit teardown custody"]
#[non_exhaustive]
pub enum NativeTextResidentClose {
    /// A planned or active use still retains this resident.
    InUse(NativeTextResident),
    /// The unique resident entered explicit native teardown.
    Teardown(NativeTextResidentTeardown),
}

/// Explicit teardown custody for a native text resident.
#[must_use = "resident teardown preserves the exact text profile and native custody"]
pub struct NativeTextResidentTeardown {
    pipeline: TextPipeline,
    inner: Box<Qwen35NativeExecutionModelTeardown>,
}

impl NativeTextResidentTeardown {
    /// Borrow the exact pipeline retained through native teardown.
    #[must_use]
    pub fn pipeline(&self) -> &TextPipeline {
        &self.pipeline
    }

    /// Return the retained native teardown state.
    #[must_use]
    pub fn state(&self) -> Qwen35NativeExecutionSessionTeardownState {
        self.inner.state()
    }

    /// Retry only an unstarted native teardown.
    pub fn retry(self) -> Self {
        Self {
            pipeline: self.pipeline,
            inner: Box::new(self.inner.retry()),
        }
    }

    /// Reconcile only native stream completion that remains unproved.
    pub fn reconcile(self) -> Self {
        Self {
            pipeline: self.pipeline,
            inner: Box::new(self.inner.reconcile()),
        }
    }
}

impl NativeTextResident {
    /// Create one shared resident from one exact text pipeline profile.
    ///
    /// # Errors
    ///
    /// Returns the exact pipeline together with a checked plan error or typed
    /// native construction failure and its partial custody.
    ///
    /// # Safety
    ///
    /// `device` must satisfy the qualified native device and compiler/math
    /// contract. The caller must separately hold current host authorization;
    /// this constructor provides neither service admission nor capacity proof.
    pub unsafe fn new(
        pipeline: TextPipeline,
        device: &Device,
        page_tokens: NativePageTokens,
    ) -> Result<Self, NativeTextResidentBuildFailure> {
        let profile = pipeline.execution_profile();
        let context_ceiling = effective_context_ceiling(profile).map_err(|source| {
            NativeTextResidentBuildFailure::Plan {
                source: Box::new(source),
                pipeline: pipeline.clone(),
                location: snafu::location!(),
            }
        })?;
        let plan = Qwen35NativeExecutionPlan::try_from_weights(
            profile.weights(),
            context_ceiling,
            page_tokens,
        )
        .map_err(|source| NativeTextResidentBuildFailure::Plan {
            source: Box::new(source),
            pipeline: pipeline.clone(),
            location: snafu::location!(),
        })?;
        // SAFETY: this boundary forwards the caller's qualified device contract.
        let model = unsafe { plan.into_model(device) }.map_err(|source| {
            NativeTextResidentBuildFailure::Native {
                source: Box::new(source),
                pipeline: pipeline.clone(),
                location: snafu::location!(),
            }
        })?;
        Ok(Self {
            inner: Arc::new(NativeTextResidentInner {
                pipeline,
                model: ExplicitReleaseOwner::new(model),
            }),
        })
    }

    /// Plan one exact prepared request without allocating a native session.
    ///
    /// An independently constructed, same-width pipeline is refused before
    /// native session planning or allocation.
    ///
    /// # Errors
    ///
    /// Refuses a foreign preparation or a context the exact resident cannot
    /// plan.
    pub fn plan_generation(
        &self,
        prepared: PreparedGeneration,
    ) -> Result<NativeTextUsePlan, NativeTextUsePlanFailure> {
        self.plan_generation_prefill(prepared, 1)
    }

    /// Plan one exact prepared request with an explicit native prefill capacity.
    ///
    /// The requested capacity is forwarded unchanged to the decoder's checked
    /// per-use plan. It therefore controls both this use's device demand and
    /// every bounded native prefill submission without changing resident state.
    ///
    /// # Errors
    ///
    /// Refuses a foreign preparation or a zero, context-exceeding, or otherwise
    /// unrepresentable native session plan before session allocation.
    pub fn plan_generation_prefill(
        &self,
        prepared: PreparedGeneration,
        max_chunk_tokens: usize,
    ) -> Result<NativeTextUsePlan, NativeTextUsePlanFailure> {
        let prepared = bind_preparation(&self.inner.pipeline, prepared)?;
        let session = self
            .inner
            .model
            .get()
            .plan_prefill_session(prepared.context_tokens(), max_chunk_tokens)
            .map_err(|source| NativeTextUsePlanFailure::NativePlan {
                source,
                location: snafu::location!(),
            })?;
        Ok(NativeTextUsePlan {
            resident: Arc::clone(&self.inner),
            recycled_logits_plan: prepared.recycled_logits_plan(),
            session,
            prepared,
        })
    }

    /// Explicitly close this resident when no planned or active use retains it.
    pub fn close(self) -> NativeTextResidentClose {
        match Arc::try_unwrap(self.inner) {
            Err(inner) => NativeTextResidentClose::InUse(Self { inner }),
            Ok(NativeTextResidentInner { pipeline, model }) => match model.into_inner().close() {
                Qwen35NativeExecutionModelClose::InUse(model) => {
                    NativeTextResidentClose::InUse(Self {
                        inner: Arc::new(NativeTextResidentInner {
                            pipeline,
                            model: ExplicitReleaseOwner::new(model),
                        }),
                    })
                }
                Qwen35NativeExecutionModelClose::Teardown(inner) => {
                    NativeTextResidentClose::Teardown(NativeTextResidentTeardown {
                        pipeline,
                        inner,
                    })
                }
            },
        }
    }
}

fn bind_preparation(
    pipeline: &TextPipeline,
    prepared: PreparedGeneration,
) -> Result<PreparedGeneration, NativeTextUsePlanFailure> {
    if pipeline.owns_preparation(&prepared) {
        return Ok(prepared);
    }
    Err(NativeTextUsePlanFailure::ForeignPreparation {
        location: snafu::location!(),
    })
}

fn effective_context_ceiling(
    profile: text::ExecutionProfile<'_>,
) -> Result<usize, decoders::Error> {
    Ok(profile
        .context_ceiling()
        .min(profile.weights().execution_context_ceiling()?))
}

/// Failure while binding one consumed text preparation to a native use plan.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum NativeTextUsePlanFailure {
    /// The preparation belongs to an independently constructed text pipeline.
    #[snafu(display("prepared generation belongs to a different native text resident pipeline"))]
    ForeignPreparation {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// The resident refused the prepared request's exact context plan.
    #[snafu(display("native text session plan failed: {source}"))]
    NativePlan {
        /// Original checked decoder-plan failure.
        source: decoders::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Opaque one-use plan retaining an exact preparation and resident.
#[must_use = "a planned use retains an exact preparation and shared resident"]
pub struct NativeTextUsePlan {
    resident: Arc<NativeTextResidentInner>,
    recycled_logits_plan: RecycledLogitsPlan,
    session: Qwen35NativeExecutionSessionPlan,
    prepared: PreparedGeneration,
}

impl NativeTextUsePlan {
    /// Return the exact host-row plan to acquire before native session creation.
    #[must_use]
    pub const fn recycled_logits_plan(&self) -> RecycledLogitsPlan {
        self.recycled_logits_plan
    }

    /// Return the exact requested device extent for this future session.
    #[must_use]
    pub const fn device_demand(&self) -> Qwen35NativeExecutionDeviceDemand {
        self.session.device_demand()
    }

    /// Return this use's exact checked B=1 prefill capacity.
    #[must_use]
    pub const fn max_chunk_tokens(&self) -> usize {
        self.session.max_chunk_tokens()
    }

    /// Execute this exact planned use with caller-acquired recycled host storage.
    ///
    /// Storage is validated and cancellation is observed before native session
    /// construction can allocate any per-use resource.
    ///
    /// Every non-final native chunk output is explicitly released before another
    /// chunk is started. Generation is published only after both final output
    /// and session teardown are acknowledged; all other outcomes retain typed
    /// custody.
    ///
    /// # Errors
    ///
    /// Returns typed storage, construction, pipeline, driver, output-release,
    /// or session-teardown custody without flattening its original source.
    ///
    /// # Safety
    ///
    /// The caller must maintain the qualified native device, compiler, and
    /// numerical contract for each submitted native chunk and separately hold current
    /// host authorization. This is not a service or admission entrypoint.
    pub unsafe fn generate(
        self,
        storage: RecycledLogitsStorage,
        cancellation: &dyn Cancellation,
    ) -> Result<Generation, NativeTextGenerationFailure> {
        if let Err(source) = self.recycled_logits_plan.validate_storage(&storage) {
            return Err(NativeTextGenerationFailure::Storage {
                source: Box::new(source),
                plan: Box::new(self),
                storage,
            });
        }
        let Self {
            resident,
            recycled_logits_plan,
            session,
            prepared,
        } = self;
        let max_chunk_tokens = session.max_chunk_tokens();
        let construction = construct_unless_cancelled(
            session,
            cancellation,
            Qwen35NativeExecutionSessionPlan::into_session,
        );
        match construction {
            ConstructionAttempt::Cancelled(session) => {
                Err(NativeTextGenerationFailure::Cancelled {
                    plan: Box::new(Self {
                        resident,
                        recycled_logits_plan,
                        session,
                        prepared,
                    }),
                    storage,
                })
            }
            ConstructionAttempt::Attempted(Err(source)) => {
                Err(NativeTextGenerationFailure::Construction {
                    source: Box::new(source),
                    custody: Box::new(NativeTextUseConstructionCustody {
                        _resident: resident,
                        _prepared: prepared,
                        _storage: storage,
                    }),
                })
            }
            ConstructionAttempt::Attempted(Ok(session)) => {
                // SAFETY: this method's contract supplies the bounded native qualification.
                unsafe {
                    generate_with_native_session(
                        resident,
                        prepared,
                        session,
                        max_chunk_tokens,
                        storage,
                        cancellation,
                    )
                }
            }
        }
    }
}

enum ConstructionAttempt<Plan, Session, Error> {
    Cancelled(Plan),
    Attempted(Result<Session, Error>),
}

fn construct_unless_cancelled<Plan, Session, Error>(
    plan: Plan,
    cancellation: &dyn Cancellation,
    construct: impl FnOnce(Plan) -> Result<Session, Error>,
) -> ConstructionAttempt<Plan, Session, Error> {
    if cancellation.is_cancelled() {
        ConstructionAttempt::Cancelled(plan)
    } else {
        ConstructionAttempt::Attempted(construct(plan))
    }
}

/// Per-use custody retained when native session construction fails.
#[must_use = "construction failure retains its exact resident, request, and host row"]
pub struct NativeTextUseConstructionCustody {
    _resident: Arc<NativeTextResidentInner>,
    _prepared: PreparedGeneration,
    _storage: RecycledLogitsStorage,
}

/// Typed native driver failure retaining output-release evidence when required.
#[must_use = "driver failures can retain unreleased native output custody"]
#[non_exhaustive]
pub enum NativeTextDriverError {
    /// One native decoder prefill submission failed.
    Native {
        /// Original decoder failure.
        source: decoders::Error,
    },
    /// Copying one final native output into the host row failed.
    Copy {
        /// Original synchronous device-to-host copy failure.
        source: hipcore::Error,
        /// Explicit release outcome for the copied-from native output.
        release: Box<BufferRelease>,
    },
    /// A native output could not be acknowledged released before reuse or publication.
    OutputRelease {
        /// Explicit release outcome retaining unresolved output custody.
        release: Box<BufferRelease>,
    },
    /// Cancellation was observed at a native chunk boundary.
    Cancelled,
    /// The shared text port violated its nonempty token-batch contract.
    EmptyTokenBatch,
}

impl fmt::Debug for NativeTextDriverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native { source } => formatter
                .debug_struct("NativeTextDriverError::Native")
                .field("source", source)
                .finish(),
            Self::Copy { source, .. } => formatter
                .debug_struct("NativeTextDriverError::Copy")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::OutputRelease { .. } => formatter
                .debug_struct("NativeTextDriverError::OutputRelease")
                .finish_non_exhaustive(),
            Self::Cancelled => formatter.write_str("NativeTextDriverError::Cancelled"),
            Self::EmptyTokenBatch => formatter.write_str("NativeTextDriverError::EmptyTokenBatch"),
        }
    }
}

impl fmt::Display for NativeTextDriverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native { source } => write!(formatter, "native text prefill failed: {source}"),
            Self::Copy { source, .. } => {
                write!(formatter, "native text logits copy failed: {source}")
            }
            Self::OutputRelease { .. } => {
                formatter.write_str("native text logits release was not acknowledged")
            }
            Self::Cancelled => formatter.write_str("native text generation was cancelled"),
            Self::EmptyTokenBatch => {
                formatter.write_str("native text driver received an empty token batch")
            }
        }
    }
}

impl std::error::Error for NativeTextDriverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Native { source } => Some(source),
            Self::Copy { source, .. } => Some(source),
            Self::OutputRelease { release } => release_error(release),
            Self::Cancelled | Self::EmptyTokenBatch => None,
        }
    }
}

impl NativeTextDriverError {
    fn retry_release(self) -> Self {
        match self {
            Self::Copy { source, release } => Self::Copy {
                source,
                release: Box::new(retry_buffer_release(*release)),
            },
            Self::OutputRelease { release } => Self::OutputRelease {
                release: Box::new(retry_buffer_release(*release)),
            },
            error => error,
        }
    }
}

enum RetryDisposition<Pending, Retained> {
    Pending(Pending),
    Retained(Retained),
}

impl<Pending, Retained> RetryDisposition<Pending, Retained> {
    fn resolve<Output>(
        self,
        retry: impl FnOnce(Pending) -> Output,
        retain: impl FnOnce(Retained) -> Output,
    ) -> Output {
        match self {
            Self::Pending(pending) => retry(pending),
            Self::Retained(retained) => retain(retained),
        }
    }
}

fn retry_buffer_release(release: BufferRelease) -> BufferRelease {
    let disposition = match release {
        BufferRelease::Pending(pending) => RetryDisposition::Pending(pending),
        release => RetryDisposition::Retained(release),
    };
    disposition.resolve(
        hipcore::PendingBufferTeardown::retry,
        core::convert::identity,
    )
}

fn retry_generation_source(
    source: RecycledGenerationError<NativeTextDriverError>,
) -> RecycledGenerationError<NativeTextDriverError> {
    match source {
        RecycledGenerationError::Driver { source } => RecycledGenerationError::Driver {
            source: source.retry_release(),
        },
        source => source,
    }
}

/// Explicit close custody for one native text use.
#[must_use = "native session close retains its resident reference and teardown evidence"]
pub struct NativeTextUseClose {
    resident: Arc<NativeTextResidentInner>,
    outcome: NativeTextUseCloseOutcome,
}

enum NativeTextUseCloseOutcome {
    Released,
    Teardown(Box<Qwen35NativeExecutionSessionTeardown>),
    Failed(Box<decoders::Error>),
}

trait ReleasedSessionTeardown: Sized {
    fn consume_released(self) -> Result<(), Box<Self>>;
}

impl ReleasedSessionTeardown for Qwen35NativeExecutionSessionTeardown {
    fn consume_released(self) -> Result<(), Box<Self>> {
        match self.into_released_model() {
            Ok(model) => {
                drop(model);
                Ok(())
            }
            Err(teardown) => Err(teardown),
        }
    }
}

enum SessionTeardownResolution<Resident, Teardown> {
    Released(Resident),
    Retained {
        resident: Resident,
        teardown: Box<Teardown>,
    },
}

fn resolve_session_teardown<Resident, Teardown>(
    resident: Resident,
    teardown: Teardown,
) -> SessionTeardownResolution<Resident, Teardown>
where
    Teardown: ReleasedSessionTeardown,
{
    match teardown.consume_released() {
        Ok(()) => SessionTeardownResolution::Released(resident),
        Err(teardown) => SessionTeardownResolution::Retained { resident, teardown },
    }
}

impl NativeTextUseClose {
    fn from_session(
        resident: Arc<NativeTextResidentInner>,
        session: Qwen35NativeExecutionSession,
    ) -> Self {
        match session.close() {
            Ok(teardown) => Self::from_teardown(resident, teardown),
            Err(source) => Self {
                resident,
                outcome: NativeTextUseCloseOutcome::Failed(Box::new(source)),
            },
        }
    }

    fn from_teardown(
        resident: Arc<NativeTextResidentInner>,
        teardown: Qwen35NativeExecutionSessionTeardown,
    ) -> Self {
        // The outer resident stays owned across consumption of the session's
        // recovered duplicate, so that duplicate can never be the last model.
        match resolve_session_teardown(resident, teardown) {
            SessionTeardownResolution::Released(resident) => Self {
                resident,
                outcome: NativeTextUseCloseOutcome::Released,
            },
            SessionTeardownResolution::Retained { resident, teardown } => Self {
                resident,
                outcome: NativeTextUseCloseOutcome::Teardown(teardown),
            },
        }
    }

    /// Return an unresolved session teardown state, if native close retained one.
    #[must_use]
    pub fn state(&self) -> Option<Qwen35NativeExecutionSessionTeardownState> {
        match &self.outcome {
            NativeTextUseCloseOutcome::Released | NativeTextUseCloseOutcome::Failed(_) => None,
            NativeTextUseCloseOutcome::Teardown(teardown) => Some(teardown.state()),
        }
    }

    fn source_error(&self) -> Option<&decoders::Error> {
        match &self.outcome {
            NativeTextUseCloseOutcome::Failed(source) => Some(source.as_ref()),
            NativeTextUseCloseOutcome::Released | NativeTextUseCloseOutcome::Teardown(_) => None,
        }
    }

    /// Retry only a native session teardown that did not begin.
    pub fn retry(self) -> Self {
        let Self { resident, outcome } = self;
        match outcome {
            NativeTextUseCloseOutcome::Teardown(teardown) => {
                Self::from_teardown(resident, teardown.retry())
            }
            outcome => Self { resident, outcome },
        }
    }

    /// Reconcile only native session completion that remains unproved.
    pub fn reconcile(self) -> Self {
        let Self { resident, outcome } = self;
        match outcome {
            NativeTextUseCloseOutcome::Teardown(teardown) => {
                Self::from_teardown(resident, teardown.reconcile())
            }
            outcome => Self { resident, outcome },
        }
    }

    fn is_released(&self) -> bool {
        matches!(self.outcome, NativeTextUseCloseOutcome::Released)
    }
}

trait CloseAcknowledgement {
    fn is_released(&self) -> bool;
}

impl CloseAcknowledgement for NativeTextUseClose {
    fn is_released(&self) -> bool {
        self.is_released()
    }
}

enum PublishAfterClose<Output, ExecutionError, Close> {
    Published(Output),
    Execution {
        source: ExecutionError,
        close: Close,
    },
    Close {
        close: Close,
    },
}

fn retain_before_publish<Output, ExecutionError, Close>(
    execution: Result<Output, ExecutionError>,
    close: Close,
) -> PublishAfterClose<Output, ExecutionError, Close>
where
    Close: CloseAcknowledgement,
{
    match execution {
        Ok(output) if close.is_released() => PublishAfterClose::Published(output),
        Ok(_) => PublishAfterClose::Close { close },
        Err(source) => PublishAfterClose::Execution { source, close },
    }
}

/// Failure after a native text use consumed its exact preparation.
#[must_use = "generation failure retains every unresolved native owner"]
#[non_exhaustive]
pub enum NativeTextGenerationFailure {
    /// Cancellation was observed before native session construction.
    Cancelled {
        /// Unallocated native use plan retained with its exact preparation.
        plan: Box<NativeTextUsePlan>,
        /// Caller-acquired row retained with the cancelled plan.
        storage: RecycledLogitsStorage,
    },
    /// The caller's row did not match this preparation before session allocation.
    Storage {
        /// Original typed text storage validation failure.
        source: Box<text::Error>,
        /// Unallocated native use plan retained with its exact preparation.
        plan: Box<NativeTextUsePlan>,
        /// Caller-acquired row retained with the rejected plan.
        storage: RecycledLogitsStorage,
    },
    /// Native session creation failed after row validation.
    Construction {
        /// Original typed native construction failure.
        source: Box<NativeBuildFailure>,
        /// Exact resident, request, and host row retained by the failed use.
        custody: Box<NativeTextUseConstructionCustody>,
    },
    /// The shared recycled text execution stopped after native session creation.
    Execution {
        /// Original text or driver failure, including future owned variants.
        source: Box<RecycledGenerationError<NativeTextDriverError>>,
        /// Explicit session close evidence retained alongside the source.
        close: NativeTextUseClose,
    },
    /// Native session close was not acknowledged after otherwise complete text generation.
    Close {
        /// Explicit session close evidence retaining the exact resident.
        close: NativeTextUseClose,
    },
}

impl NativeTextGenerationFailure {
    /// Retry only pending output or session releases while retaining the source.
    pub fn retry(self) -> Self {
        match self {
            Self::Execution { source, close } => Self::Execution {
                source: Box::new(retry_generation_source(*source)),
                close: close.retry(),
            },
            Self::Close { close } => Self::Close {
                close: close.retry(),
            },
            failure => failure,
        }
    }

    /// Reconcile only session completion that remains unproved.
    pub fn reconcile(self) -> Self {
        match self {
            Self::Execution { source, close } => Self::Execution {
                source,
                close: close.reconcile(),
            },
            Self::Close { close } => Self::Close {
                close: close.reconcile(),
            },
            failure => failure,
        }
    }

    /// Return the unresolved session teardown state, if any.
    #[must_use]
    pub fn session_teardown_state(&self) -> Option<Qwen35NativeExecutionSessionTeardownState> {
        match self {
            Self::Execution { close, .. } | Self::Close { close } => close.state(),
            Self::Cancelled { .. } | Self::Storage { .. } | Self::Construction { .. } => None,
        }
    }
}

impl fmt::Debug for NativeTextGenerationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Cancelled")
                .finish_non_exhaustive(),
            Self::Storage { source, .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Storage")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::Construction { source, .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Construction")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::Execution { source, .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Execution")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::Close { .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Close")
                .finish_non_exhaustive(),
        }
    }
}

impl fmt::Display for NativeTextGenerationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { .. } => formatter
                .write_str("native text generation was cancelled before session construction"),
            Self::Storage { source, .. } => {
                write!(formatter, "native text row validation failed: {source}")
            }
            Self::Construction { source, .. } => {
                write!(
                    formatter,
                    "native text session construction failed: {source}"
                )
            }
            Self::Execution { source, .. } => {
                write!(formatter, "native text generation failed: {source}")
            }
            Self::Close { close } => match close.source_error() {
                Some(source) => write!(formatter, "native text session close failed: {source}"),
                None => formatter.write_str("native text session close was not acknowledged"),
            },
        }
    }
}

impl std::error::Error for NativeTextGenerationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cancelled { .. } => None,
            Self::Storage { source, .. } => Some(source.as_ref()),
            Self::Construction { source, .. } => Some(source.as_ref()),
            Self::Execution { source, .. } => Some(source.as_ref()),
            Self::Close { close } => close
                .source_error()
                .map(|source| source as &(dyn std::error::Error + 'static)),
        }
    }
}

struct NativeTextDriver {
    session: Qwen35NativeExecutionSession,
    // INVARIANT: the checked native session plan is the sole authority for this
    // positive capacity; this driver neither clamps nor supplies a default.
    max_chunk_tokens: usize,
}

trait NativeTokenDriver {
    type Output;

    fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<Self::Output, NativeTextDriverError>;

    fn release_intermediate(&mut self, output: Self::Output) -> Result<(), NativeTextDriverError>;

    fn copy_and_release_final(
        &mut self,
        output: Self::Output,
        logits: &mut [f32],
    ) -> Result<(), NativeTextDriverError>;
}

impl NativeTokenDriver for NativeTextDriver {
    type Output = DeviceBuffer<f32>;

    fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<Self::Output, NativeTextDriverError> {
        // SAFETY: `NativeTextUsePlan::generate` establishes this native execution contract.
        unsafe { self.session.prefill(tokens) }
            .map_err(|source| NativeTextDriverError::Native { source })
    }

    fn release_intermediate(&mut self, output: Self::Output) -> Result<(), NativeTextDriverError> {
        release_intermediate(output)
    }

    fn copy_and_release_final(
        &mut self,
        output: Self::Output,
        logits: &mut [f32],
    ) -> Result<(), NativeTextDriverError> {
        copy_and_release_final(output, logits)
    }
}

impl RecycledGenerationDriver for NativeTextDriver {
    type Error = NativeTextDriverError;

    fn step_into(
        &mut self,
        token_ids: &[u32],
        logits: &mut [f32],
        cancellation: &dyn Cancellation,
    ) -> Result<(), Self::Error> {
        let max_chunk_tokens = self.max_chunk_tokens;
        drive_token_batch(self, max_chunk_tokens, token_ids, logits, cancellation)
    }
}

fn drive_token_batch<Driver>(
    driver: &mut Driver,
    max_chunk_tokens: usize,
    token_ids: &[u32],
    logits: &mut [f32],
    cancellation: &dyn Cancellation,
) -> Result<(), NativeTextDriverError>
where
    Driver: NativeTokenDriver,
{
    if token_ids.is_empty() {
        return Err(NativeTextDriverError::EmptyTokenBatch);
    }
    let mut chunks = token_ids.chunks(max_chunk_tokens).peekable();
    while let Some(tokens) = chunks.next() {
        if cancellation.is_cancelled() {
            return Err(NativeTextDriverError::Cancelled);
        }
        let output = driver.prefill_tokens(tokens)?;
        if chunks.peek().is_some() {
            driver.release_intermediate(output)?;
        } else {
            return driver.copy_and_release_final(output, logits);
        }
    }
    Err(NativeTextDriverError::EmptyTokenBatch)
}

unsafe fn generate_with_native_session(
    resident: Arc<NativeTextResidentInner>,
    prepared: PreparedGeneration,
    session: Qwen35NativeExecutionSession,
    max_chunk_tokens: usize,
    storage: RecycledLogitsStorage,
    cancellation: &dyn Cancellation,
) -> Result<Generation, NativeTextGenerationFailure> {
    let mut driver = NativeTextDriver {
        session,
        max_chunk_tokens,
    };
    let generation = prepared.generate_with_recycled_driver(&mut driver, storage, cancellation);
    let close = NativeTextUseClose::from_session(resident, driver.session);
    finish_generation(generation, close)
}

fn finish_generation(
    generation: Result<Generation, RecycledGenerationError<NativeTextDriverError>>,
    close: NativeTextUseClose,
) -> Result<Generation, NativeTextGenerationFailure> {
    match retain_before_publish(generation, close) {
        PublishAfterClose::Published(generation) => Ok(generation),
        PublishAfterClose::Execution { source, close } => {
            Err(NativeTextGenerationFailure::Execution {
                source: Box::new(source),
                close,
            })
        }
        PublishAfterClose::Close { close } => Err(NativeTextGenerationFailure::Close { close }),
    }
}

fn release_intermediate(output: DeviceBuffer<f32>) -> Result<(), NativeTextDriverError> {
    match output.begin_release() {
        BufferRelease::Released(_) => Ok(()),
        release => Err(NativeTextDriverError::OutputRelease {
            release: Box::new(release),
        }),
    }
}

fn copy_and_release_final(
    output: DeviceBuffer<f32>,
    logits: &mut [f32],
) -> Result<(), NativeTextDriverError> {
    if let Err(source) = output.copy_to_host(logits) {
        return Err(NativeTextDriverError::Copy {
            source,
            release: Box::new(output.begin_release()),
        });
    }
    match output.begin_release() {
        BufferRelease::Released(_) => Ok(()),
        release => Err(NativeTextDriverError::OutputRelease {
            release: Box::new(release),
        }),
    }
}

fn release_error(release: &BufferRelease) -> Option<&(dyn std::error::Error + 'static)> {
    match release {
        BufferRelease::Pending(pending) => Some(pending.error()),
        BufferRelease::Quarantined(quarantine) => Some(quarantine.error()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use loader::gguf::{ArtifactByteLimit, Sha256Digest, VerifiedArtifact};
    use sha2::{Digest, Sha256};
    use test_fixtures::{Qwen35FixtureConfig, build_qwen35_fixture};
    use text::{
        GenerationRequest, NeverCancelled, PipelineLimits, TextMessage, TextRole,
        TokenizerCompanion,
    };
    use tokenize::{TokenizerByteLimit, TokenizerDigest, TokenizerIdentity};

    use super::*;

    const TOKENIZER_JSON: &str = r#"{
      "version":"1.0","truncation":null,"padding":null,
      "added_tokens":[
        {"id":1,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        {"id":2,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
      ],
      "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
      "post_processor":null,"decoder":null,
      "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<bos>":1,"<eos>":2,"hello":3,"assistant":4},"unk_token":"[UNK]"}
    }"#;

    fn synthetic_pipeline(
        context_tokens: usize,
    ) -> Result<(tempfile::TempDir, TextPipeline), Box<dyn std::error::Error>> {
        let fixture = build_qwen35_fixture(&Qwen35FixtureConfig {
            chat_template: "{{ messages[0].content }}".to_owned(),
            ..Qwen35FixtureConfig::default()
        })?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("synthetic.gguf");
        std::fs::write(&path, fixture.bytes)?;
        let artifact =
            VerifiedArtifact::load(
                &path,
                Sha256Digest::from_bytes(fixture.sha256),
                ArtifactByteLimit::new(NonZeroU64::new(fixture.byte_len).ok_or_else(|| {
                    std::io::Error::other("synthetic fixture has zero byte length")
                })?),
            )?;
        let tokenizer_bytes = TOKENIZER_JSON.as_bytes();
        let tokenizer_limit = TokenizerByteLimit::new(
            NonZeroUsize::new(tokenizer_bytes.len())
                .ok_or_else(|| std::io::Error::other("synthetic tokenizer is empty"))?,
        );
        let digest = TokenizerDigest::from_bytes(Sha256::digest(tokenizer_bytes).into());
        let pipeline = TextPipeline::new(
            &artifact,
            TokenizerCompanion::new(
                tokenizer_bytes,
                TokenizerIdentity::new(tokenizer_bytes.len(), digest),
            ),
            PipelineLimits {
                tokenizer_bytes: tokenizer_limit,
                template_bytes: 4_096,
                messages: 4,
                message_bytes: 128,
                prompt_bytes: 256,
                rendered_bytes: 256,
                context_tokens,
                output_tokens: 6,
                output_bytes: 128,
                template_fuel: 10_000,
                template_recursion: 16,
            },
        )?;
        Ok((directory, pipeline))
    }

    fn prepared(pipeline: &TextPipeline) -> Result<PreparedGeneration, text::Error> {
        let messages = [TextMessage::new(TextRole::User, "hello")];
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)
    }

    #[derive(Default)]
    struct SyntheticDriver {
        prefills: Vec<Vec<u32>>,
        released: Vec<u32>,
        unresolved_output: Option<u32>,
        final_output: Option<u32>,
        fails_on: Option<u32>,
        release_fails_on: Option<u32>,
    }

    impl NativeTokenDriver for SyntheticDriver {
        type Output = u32;

        fn prefill_tokens(
            &mut self,
            tokens: &[u32],
        ) -> Result<Self::Output, NativeTextDriverError> {
            self.prefills.push(tokens.to_vec());
            if self.fails_on.is_some_and(|token| tokens.contains(&token)) {
                return Err(NativeTextDriverError::EmptyTokenBatch);
            }
            tokens
                .last()
                .copied()
                .ok_or(NativeTextDriverError::EmptyTokenBatch)
        }

        fn release_intermediate(
            &mut self,
            output: Self::Output,
        ) -> Result<(), NativeTextDriverError> {
            if self.release_fails_on == Some(output) {
                self.unresolved_output = Some(output);
                return Err(NativeTextDriverError::EmptyTokenBatch);
            }
            self.released.push(output);
            Ok(())
        }

        fn copy_and_release_final(
            &mut self,
            output: Self::Output,
            logits: &mut [f32],
        ) -> Result<(), NativeTextDriverError> {
            self.final_output = Some(output);
            logits[0] = f32::from(
                u16::try_from(output).map_err(|_| NativeTextDriverError::EmptyTokenBatch)?,
            );
            Ok(())
        }
    }

    struct CancelAt {
        call: Cell<usize>,
        cancelled_call: usize,
    }

    impl Cancellation for CancelAt {
        fn is_cancelled(&self) -> bool {
            let call = self.call.get();
            self.call.set(call + 1);
            call == self.cancelled_call
        }
    }

    struct DropProbe<'a> {
        drops: &'a AtomicUsize,
    }

    impl Drop for DropProbe<'_> {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct SyntheticReleasedTeardown<'a> {
        duplicate: DropProbe<'a>,
    }

    impl ReleasedSessionTeardown for SyntheticReleasedTeardown<'_> {
        fn consume_released(self) -> Result<(), Box<Self>> {
            drop(self.duplicate);
            Ok(())
        }
    }

    struct SyntheticClose<'a> {
        released: bool,
        drops: &'a AtomicUsize,
    }

    impl CloseAcknowledgement for SyntheticClose<'_> {
        fn is_released(&self) -> bool {
            self.released
        }
    }

    impl Drop for SyntheticClose<'_> {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[derive(Debug, Eq, PartialEq)]
    enum SyntheticRelease {
        Retried,
        Quarantined,
        Future,
    }

    #[test]
    fn exact_pipeline_binding_refuses_a_sibling_preparation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_first_directory, first) = synthetic_pipeline(8)?;
        let (_second_directory, second) = synthetic_pipeline(8)?;

        assert!(bind_preparation(&first, prepared(&first)?).is_ok());
        assert!(matches!(
            bind_preparation(&first, prepared(&second)?),
            Err(NativeTextUsePlanFailure::ForeignPreparation { .. })
        ));
        Ok(())
    }

    #[test]
    fn prefill_capacity_refuses_zero_and_context_excess_before_native_allocation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, pipeline) = synthetic_pipeline(8)?;
        let profile = pipeline.execution_profile();

        assert!(
            Qwen35NativeExecutionPlan::try_from_weights_prefill(
                profile.weights(),
                8,
                0,
                NativePageTokens::B8,
            )
            .is_err()
        );
        assert!(
            Qwen35NativeExecutionPlan::try_from_weights_prefill(
                profile.weights(),
                8,
                9,
                NativePageTokens::B8,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn preconstruction_cancellation_does_not_invoke_the_constructor()
    -> Result<(), Box<dyn std::error::Error>> {
        let calls = Cell::new(0);
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: 0,
        };
        let attempt = construct_unless_cancelled(17_u32, &cancellation, |plan| {
            calls.set(calls.get() + 1);
            Ok::<u32, ()>(plan)
        });
        let ConstructionAttempt::Cancelled(plan) = attempt else {
            return Err(std::io::Error::other("cancelled construction was attempted").into());
        };

        assert_eq!(plan, 17);
        assert_eq!(calls.get(), 0);
        assert_eq!(cancellation.call.get(), 1);
        Ok(())
    }

    #[test]
    fn released_session_duplicate_is_consumed_while_resident_is_retained()
    -> Result<(), Box<dyn std::error::Error>> {
        let resident_drops = AtomicUsize::new(0);
        let duplicate_drops = AtomicUsize::new(0);
        let resolution = resolve_session_teardown(
            DropProbe {
                drops: &resident_drops,
            },
            SyntheticReleasedTeardown {
                duplicate: DropProbe {
                    drops: &duplicate_drops,
                },
            },
        );

        assert_eq!(duplicate_drops.load(Ordering::Relaxed), 1);
        assert_eq!(resident_drops.load(Ordering::Relaxed), 0);
        let SessionTeardownResolution::Released(resident) = resolution else {
            return Err(std::io::Error::other("released teardown remained retained").into());
        };
        drop(resident);
        assert_eq!(resident_drops.load(Ordering::Relaxed), 1);
        assert_eq!(duplicate_drops.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn nonreleased_close_withholds_output_and_retains_close()
    -> Result<(), Box<dyn std::error::Error>> {
        let output_drops = AtomicUsize::new(0);
        let close_drops = AtomicUsize::new(0);
        let result = retain_before_publish::<_, (), _>(
            Ok(DropProbe {
                drops: &output_drops,
            }),
            SyntheticClose {
                released: false,
                drops: &close_drops,
            },
        );

        assert_eq!(output_drops.load(Ordering::Relaxed), 1);
        assert_eq!(close_drops.load(Ordering::Relaxed), 0);
        let PublishAfterClose::Close { close } = result else {
            return Err(std::io::Error::other("nonreleased close published output").into());
        };
        drop(close);
        assert_eq!(close_drops.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn retry_transition_invokes_only_pending_release() {
        let retries = Cell::new(0);
        let retry = |()| {
            retries.set(retries.get() + 1);
            SyntheticRelease::Retried
        };

        let pending = RetryDisposition::<(), SyntheticRelease>::Pending(());
        assert_eq!(
            pending.resolve(retry, core::convert::identity),
            SyntheticRelease::Retried
        );
        let quarantine = RetryDisposition::<(), _>::Retained(SyntheticRelease::Quarantined);
        assert_eq!(
            quarantine.resolve(retry, core::convert::identity),
            SyntheticRelease::Quarantined
        );
        let future = RetryDisposition::<(), _>::Retained(SyntheticRelease::Future);
        assert_eq!(
            future.resolve(retry, core::convert::identity),
            SyntheticRelease::Future
        );
        assert_eq!(retries.get(), 1);
    }

    #[test]
    fn shared_explicit_owner_never_drops_on_arc_abandonment() {
        let abandoned_drops = AtomicUsize::new(0);
        let owner = Arc::new(ExplicitReleaseOwner::new(DropProbe {
            drops: &abandoned_drops,
        }));
        let sibling = Arc::clone(&owner);

        drop(owner);
        assert_eq!(abandoned_drops.load(Ordering::Relaxed), 0);
        drop(sibling);
        assert_eq!(abandoned_drops.load(Ordering::Relaxed), 0);

        let released_drops = AtomicUsize::new(0);
        let released = ExplicitReleaseOwner::new(DropProbe {
            drops: &released_drops,
        });
        drop(released.into_inner());
        assert_eq!(released_drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn capacity_one_prefill_preserves_token_serial_execution() {
        let mut driver = SyntheticDriver::default();
        let mut logits = [0.0];
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: usize::MAX,
        };

        let result = drive_token_batch(&mut driver, 1, &[11, 12, 13], &mut logits, &cancellation);

        assert!(result.is_ok());
        assert_eq!(driver.prefills, [vec![11], vec![12], vec![13]]);
        assert_eq!(driver.released, [11, 12]);
        assert_eq!(driver.final_output, Some(13));
        assert_eq!(logits.map(f32::to_bits), [13.0_f32.to_bits()]);
    }

    #[test]
    fn uneven_prefill_chunks_release_every_nonfinal_output() {
        let mut driver = SyntheticDriver::default();
        let mut logits = [0.0];
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: usize::MAX,
        };

        let result = drive_token_batch(
            &mut driver,
            2,
            &[29, 30, 31, 32, 33],
            &mut logits,
            &cancellation,
        );

        assert!(result.is_ok());
        assert_eq!(driver.prefills, [vec![29, 30], vec![31, 32], vec![33]]);
        assert_eq!(driver.released, [30, 32]);
        assert_eq!(driver.final_output, Some(33));
        assert_eq!(logits.map(f32::to_bits), [33.0_f32.to_bits()]);
    }

    #[test]
    fn continuation_uses_one_final_prefill_output() {
        let mut driver = SyntheticDriver::default();
        let mut logits = [0.0];
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: usize::MAX,
        };

        let result = drive_token_batch(&mut driver, 3, &[29], &mut logits, &cancellation);

        assert!(result.is_ok());
        assert_eq!(driver.prefills, [vec![29]]);
        assert!(driver.released.is_empty());
        assert_eq!(driver.final_output, Some(29));
    }

    #[test]
    fn cancellation_before_prefill_submits_no_chunk() {
        let mut driver = SyntheticDriver::default();
        let mut logits = [0.0];
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: 0,
        };

        let result = drive_token_batch(&mut driver, 2, &[41, 42, 43], &mut logits, &cancellation);

        assert!(matches!(result, Err(NativeTextDriverError::Cancelled)));
        assert!(driver.prefills.is_empty());
        assert!(driver.released.is_empty());
        assert_eq!(driver.final_output, None);
    }

    #[test]
    fn cancellation_between_prefill_chunks_preserves_prior_release() {
        let mut driver = SyntheticDriver::default();
        let mut logits = [0.0];
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: 1,
        };

        let result = drive_token_batch(
            &mut driver,
            2,
            &[41, 42, 43, 44, 45],
            &mut logits,
            &cancellation,
        );

        assert!(matches!(result, Err(NativeTextDriverError::Cancelled)));
        assert_eq!(driver.prefills, [vec![41, 42]]);
        assert_eq!(driver.released, [42]);
        assert_eq!(driver.final_output, None);
    }

    #[test]
    fn native_failure_keeps_later_outputs_unvisited() {
        let mut driver = SyntheticDriver {
            fails_on: Some(53),
            ..SyntheticDriver::default()
        };
        let mut logits = [0.0];
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: usize::MAX,
        };

        let result = drive_token_batch(
            &mut driver,
            2,
            &[51, 52, 53, 54, 55],
            &mut logits,
            &cancellation,
        );

        assert!(matches!(
            result,
            Err(NativeTextDriverError::EmptyTokenBatch)
        ));
        assert_eq!(driver.prefills, [vec![51, 52], vec![53, 54]]);
        assert_eq!(driver.released, [52]);
        assert_eq!(driver.final_output, None);
    }

    #[test]
    fn intermediate_release_failure_returns_original_error_without_later_driver_actions() {
        let mut driver = SyntheticDriver {
            release_fails_on: Some(62),
            ..SyntheticDriver::default()
        };
        let mut logits = [0.0];
        let cancellation = CancelAt {
            call: Cell::new(0),
            cancelled_call: usize::MAX,
        };

        let result = drive_token_batch(
            &mut driver,
            2,
            &[61, 62, 63, 64, 65],
            &mut logits,
            &cancellation,
        );

        assert!(matches!(result, Err(NativeTextDriverError::EmptyTokenBatch)));
        assert_eq!(driver.prefills, [vec![61, 62]]);
        assert!(driver.released.is_empty());
        assert_eq!(driver.unresolved_output, Some(62));
        assert_eq!(driver.final_output, None);
        assert_eq!(logits, [0.0]);
    }
}
