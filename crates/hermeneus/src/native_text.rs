//! Shared native text execution with explicit per-use custody.

use std::fmt;
use std::sync::Arc;

use decoders::{
    NativeBuildFailure, Qwen35NativeExecutionDeviceDemand, Qwen35NativeExecutionModel,
    Qwen35NativeExecutionModelClose, Qwen35NativeExecutionModelTeardown, Qwen35NativeExecutionPlan,
    Qwen35NativeExecutionSession, Qwen35NativeExecutionSessionPlan,
    Qwen35NativeExecutionSessionTeardown, Qwen35NativeExecutionSessionTeardownState,
};
use hipcore::{BufferRelease, Device, DeviceBuffer};
use kernels::attention::NativePageTokens;
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
    model: Qwen35NativeExecutionModel,
}

/// Failure while creating a shared native text resident.
#[must_use = "native construction failure retains its exact typed source"]
pub enum NativeTextResidentBuildFailure {
    /// The exact pipeline profile could not form a native execution plan.
    Plan {
        /// Original checked decoder-plan failure.
        source: decoders::Error,
        /// Exact text pipeline retained by the failed construction.
        pipeline: TextPipeline,
    },
    /// Native creation failed after retaining typed partial native custody.
    Native {
        /// Original typed native construction custody.
        source: Box<NativeBuildFailure>,
        /// Exact text pipeline retained by the failed construction.
        pipeline: TextPipeline,
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

impl fmt::Display for NativeTextResidentBuildFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan { source, .. } => write!(formatter, "native text plan failed: {source}"),
            Self::Native { source, .. } => {
                write!(
                    formatter,
                    "native text resident construction failed: {source}"
                )
            }
        }
    }
}

impl std::error::Error for NativeTextResidentBuildFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plan { source, .. } => Some(source),
            Self::Native { source, .. } => Some(source.as_ref()),
        }
    }
}

/// Result of explicitly closing a native text resident.
#[must_use = "resident close retains the model or its explicit teardown custody"]
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
    #[must_use]
    pub fn retry(self) -> Self {
        Self {
            pipeline: self.pipeline,
            inner: Box::new(self.inner.retry()),
        }
    }

    /// Reconcile only native stream completion that remains unproved.
    #[must_use]
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
        let artifact_ceiling = profile
            .weights()
            .execution_context_ceiling()
            .map_err(|source| NativeTextResidentBuildFailure::Plan {
                source,
                pipeline: pipeline.clone(),
            })?;
        let context_ceiling = profile.context_ceiling().min(artifact_ceiling);
        let plan = Qwen35NativeExecutionPlan::try_from_weights(
            profile.weights(),
            context_ceiling,
            page_tokens,
        )
        .map_err(|source| NativeTextResidentBuildFailure::Plan {
            source,
            pipeline: pipeline.clone(),
        })?;
        // SAFETY: this boundary forwards the caller's qualified device contract.
        let model = unsafe { plan.into_model(device) }.map_err(|source| {
            NativeTextResidentBuildFailure::Native {
                source: Box::new(source),
                pipeline: pipeline.clone(),
            }
        })?;
        Ok(Self {
            inner: Arc::new(NativeTextResidentInner { pipeline, model }),
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
        if !self.inner.pipeline.owns_preparation(&prepared) {
            return Err(NativeTextUsePlanFailure::ForeignPreparation);
        }
        let session = self
            .inner
            .model
            .plan_session(prepared.context_tokens())
            .map_err(|source| NativeTextUsePlanFailure::NativePlan { source })?;
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
            Ok(NativeTextResidentInner { pipeline, model }) => match model.close() {
                Qwen35NativeExecutionModelClose::InUse(model) => {
                    NativeTextResidentClose::InUse(Self {
                        inner: Arc::new(NativeTextResidentInner { pipeline, model }),
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

/// Failure while binding one consumed text preparation to a native use plan.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativeTextUsePlanFailure {
    /// The preparation belongs to an independently constructed text pipeline.
    ForeignPreparation,
    /// The resident refused the prepared request's exact context plan.
    NativePlan {
        /// Original checked decoder-plan failure.
        source: decoders::Error,
    },
}

impl fmt::Display for NativeTextUsePlanFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignPreparation => formatter.write_str(
                "prepared generation belongs to a different native text resident pipeline",
            ),
            Self::NativePlan { source } => {
                write!(formatter, "native text session plan failed: {source}")
            }
        }
    }
}

impl std::error::Error for NativeTextUsePlanFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ForeignPreparation => None,
            Self::NativePlan { source } => Some(source),
        }
    }
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

    /// Execute this exact planned use with caller-acquired recycled host storage.
    ///
    /// Every native token output is explicitly released before another token is
    /// started. Generation is published only after both final output and session
    /// teardown are acknowledged; all other outcomes retain typed custody.
    ///
    /// # Errors
    ///
    /// Returns typed storage, construction, pipeline, driver, output-release,
    /// or session-teardown custody without flattening its original source.
    ///
    /// # Safety
    ///
    /// The caller must maintain the qualified native device, compiler, and
    /// numerical contract for each submitted token and separately hold current
    /// host authorization. This is not a service or admission entrypoint.
    pub unsafe fn generate(
        self,
        storage: RecycledLogitsStorage,
        cancellation: &dyn Cancellation,
    ) -> Result<Generation, NativeTextGenerationFailure> {
        if let Err(source) = self.recycled_logits_plan.validate_storage(&storage) {
            return Err(NativeTextGenerationFailure::Storage {
                source,
                plan: Box::new(self),
                storage,
            });
        }
        let Self {
            resident,
            recycled_logits_plan: _,
            session,
            prepared,
        } = self;
        let session =
            session
                .into_session()
                .map_err(|source| NativeTextGenerationFailure::Construction {
                    source: Box::new(source),
                    custody: NativeTextUseConstructionCustody {
                        _resident: resident,
                        _prepared: prepared,
                        _storage: storage,
                    },
                })?;
        // SAFETY: this method's contract supplies the per-token native qualification.
        unsafe { generate_with_native_session(resident, prepared, session, storage, cancellation) }
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
    /// One native decoder token step failed.
    Native {
        /// Original decoder failure.
        source: decoders::Error,
    },
    /// Copying one final native output into the host row failed.
    Copy {
        /// Original synchronous device-to-host copy failure.
        source: hipcore::Error,
        /// Explicit release outcome for the copied-from native output.
        release: BufferRelease,
    },
    /// A native output could not be acknowledged released before reuse or publication.
    OutputRelease {
        /// Explicit release outcome retaining unresolved output custody.
        release: BufferRelease,
    },
    /// Cancellation was observed between complete native prompt-token operations.
    Cancelled,
    /// The shared text port violated its nonempty token-batch contract.
    EmptyTokenBatch,
}

impl NativeTextDriverError {
    fn retry_release(self) -> Self {
        match self {
            Self::Copy { source, release } => Self::Copy {
                source,
                release: retry_buffer_release(release),
            },
            Self::OutputRelease { release } => Self::OutputRelease {
                release: retry_buffer_release(release),
            },
            error => error,
        }
    }
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
            Self::Native { source } => write!(formatter, "native text token step failed: {source}"),
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

/// Explicit close custody for one native text use.
#[must_use = "native session close retains its resident reference and teardown evidence"]
pub struct NativeTextUseClose {
    resident: Arc<NativeTextResidentInner>,
    outcome: NativeTextUseCloseOutcome,
}

enum NativeTextUseCloseOutcome {
    Released,
    Teardown(Box<Qwen35NativeExecutionSessionTeardown>),
    Failed(decoders::Error),
}

impl NativeTextUseClose {
    fn from_session(
        resident: Arc<NativeTextResidentInner>,
        session: Qwen35NativeExecutionSession,
    ) -> Self {
        match session.close() {
            Ok(teardown)
                if teardown.state() == Qwen35NativeExecutionSessionTeardownState::Released =>
            {
                Self {
                    resident,
                    outcome: NativeTextUseCloseOutcome::Released,
                }
            }
            Ok(teardown) => Self {
                resident,
                outcome: NativeTextUseCloseOutcome::Teardown(Box::new(teardown)),
            },
            Err(source) => Self {
                resident,
                outcome: NativeTextUseCloseOutcome::Failed(source),
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
            NativeTextUseCloseOutcome::Failed(source) => Some(source),
            NativeTextUseCloseOutcome::Released | NativeTextUseCloseOutcome::Teardown(_) => None,
        }
    }

    /// Retry only a native session teardown that did not begin.
    #[must_use]
    pub fn retry(self) -> Self {
        let Self { resident, outcome } = self;
        let outcome = match outcome {
            NativeTextUseCloseOutcome::Teardown(teardown) => {
                NativeTextUseCloseOutcome::Teardown(Box::new(teardown.retry()))
            }
            outcome => outcome,
        };
        Self { resident, outcome }
    }

    /// Reconcile only native session completion that remains unproved.
    #[must_use]
    pub fn reconcile(self) -> Self {
        let Self { resident, outcome } = self;
        let outcome = match outcome {
            NativeTextUseCloseOutcome::Teardown(teardown) => {
                NativeTextUseCloseOutcome::Teardown(Box::new(teardown.reconcile()))
            }
            outcome => outcome,
        };
        Self { resident, outcome }
    }

    fn is_released(&self) -> bool {
        matches!(self.outcome, NativeTextUseCloseOutcome::Released)
    }
}

/// Failure after a native text use consumed its exact preparation.
#[must_use = "generation failure retains every unresolved native owner"]
pub enum NativeTextGenerationFailure {
    /// The caller's row did not match this preparation before session allocation.
    Storage {
        /// Original typed text storage validation failure.
        source: text::Error,
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
        custody: NativeTextUseConstructionCustody,
    },
    /// The shared text pipeline stopped after native session creation.
    Pipeline {
        /// Original typed text pipeline failure.
        source: text::Error,
        /// Explicit session close evidence retained alongside the source.
        close: NativeTextUseClose,
    },
    /// The native driver stopped after native session creation.
    Driver {
        /// Original typed native driver failure.
        source: NativeTextDriverError,
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
    #[must_use]
    pub fn retry(self) -> Self {
        match self {
            Self::Driver { source, close } => Self::Driver {
                source: source.retry_release(),
                close: close.retry(),
            },
            Self::Pipeline { source, close } => Self::Pipeline {
                source,
                close: close.retry(),
            },
            Self::Close { close } => Self::Close {
                close: close.retry(),
            },
            failure => failure,
        }
    }

    /// Reconcile only session completion that remains unproved.
    #[must_use]
    pub fn reconcile(self) -> Self {
        match self {
            Self::Driver { source, close } => Self::Driver {
                source,
                close: close.reconcile(),
            },
            Self::Pipeline { source, close } => Self::Pipeline {
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
            Self::Pipeline { close, .. } | Self::Driver { close, .. } | Self::Close { close } => {
                close.state()
            }
            Self::Storage { .. } | Self::Construction { .. } => None,
        }
    }
}

impl fmt::Debug for NativeTextGenerationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage { source, .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Storage")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::Construction { source, .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Construction")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::Pipeline { source, .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Pipeline")
                .field("source", source)
                .finish_non_exhaustive(),
            Self::Driver { source, .. } => formatter
                .debug_struct("NativeTextGenerationFailure::Driver")
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
            Self::Storage { source, .. } => {
                write!(formatter, "native text row validation failed: {source}")
            }
            Self::Construction { source, .. } => {
                write!(
                    formatter,
                    "native text session construction failed: {source}"
                )
            }
            Self::Pipeline { source, .. } => {
                write!(formatter, "native text pipeline failed: {source}")
            }
            Self::Driver { source, .. } => write!(formatter, "native text driver failed: {source}"),
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
            Self::Storage { source, .. } | Self::Pipeline { source, .. } => Some(source),
            Self::Construction { source, .. } => Some(source.as_ref()),
            Self::Driver { source, .. } => Some(source),
            Self::Close { close } => close
                .source_error()
                .map(|source| source as &(dyn std::error::Error + 'static)),
        }
    }
}

struct NativeTextDriver {
    session: Qwen35NativeExecutionSession,
}

trait NativeTokenDriver {
    type Output;

    fn step_token(&mut self, token: u32) -> Result<Self::Output, NativeTextDriverError>;

    fn release_intermediate(&mut self, output: Self::Output) -> Result<(), NativeTextDriverError>;

    fn copy_and_release_final(
        &mut self,
        output: Self::Output,
        logits: &mut [f32],
    ) -> Result<(), NativeTextDriverError>;
}

impl NativeTokenDriver for NativeTextDriver {
    type Output = DeviceBuffer<f32>;

    fn step_token(&mut self, token: u32) -> Result<Self::Output, NativeTextDriverError> {
        // SAFETY: `NativeTextUsePlan::generate` establishes this native execution contract.
        unsafe { self.session.step(token) }
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
        drive_token_batch(self, token_ids, logits, cancellation)
    }
}

fn drive_token_batch<Driver>(
    driver: &mut Driver,
    token_ids: &[u32],
    logits: &mut [f32],
    cancellation: &dyn Cancellation,
) -> Result<(), NativeTextDriverError>
where
    Driver: NativeTokenDriver,
{
    let Some((last, prefix)) = token_ids.split_last() else {
        return Err(NativeTextDriverError::EmptyTokenBatch);
    };
    for token in prefix {
        if cancellation.is_cancelled() {
            return Err(NativeTextDriverError::Cancelled);
        }
        let output = driver.step_token(*token)?;
        driver.release_intermediate(output)?;
    }
    if cancellation.is_cancelled() {
        return Err(NativeTextDriverError::Cancelled);
    }
    let output = driver.step_token(*last)?;
    driver.copy_and_release_final(output, logits)
}

unsafe fn generate_with_native_session(
    resident: Arc<NativeTextResidentInner>,
    prepared: PreparedGeneration,
    session: Qwen35NativeExecutionSession,
    storage: RecycledLogitsStorage,
    cancellation: &dyn Cancellation,
) -> Result<Generation, NativeTextGenerationFailure> {
    let mut driver = NativeTextDriver { session };
    let generation = prepared.generate_with_recycled_driver(&mut driver, storage, cancellation);
    let close = NativeTextUseClose::from_session(resident, driver.session);
    finish_generation(generation, close)
}

fn finish_generation(
    generation: Result<Generation, RecycledGenerationError<NativeTextDriverError>>,
    close: NativeTextUseClose,
) -> Result<Generation, NativeTextGenerationFailure> {
    match generation {
        Ok(generation) if close.is_released() => Ok(generation),
        Ok(_) => Err(NativeTextGenerationFailure::Close { close }),
        Err(RecycledGenerationError::Pipeline { source }) => {
            Err(NativeTextGenerationFailure::Pipeline { source, close })
        }
        Err(RecycledGenerationError::Driver { source }) => {
            Err(NativeTextGenerationFailure::Driver { source, close })
        }
    }
}

fn release_intermediate(output: DeviceBuffer<f32>) -> Result<(), NativeTextDriverError> {
    match output.begin_release() {
        BufferRelease::Released(_) => Ok(()),
        release => Err(NativeTextDriverError::OutputRelease { release }),
    }
}

fn copy_and_release_final(
    output: DeviceBuffer<f32>,
    logits: &mut [f32],
) -> Result<(), NativeTextDriverError> {
    if let Err(source) = output.copy_to_host(logits) {
        return Err(NativeTextDriverError::Copy {
            source,
            release: output.begin_release(),
        });
    }
    match output.begin_release() {
        BufferRelease::Released(_) => Ok(()),
        release => Err(NativeTextDriverError::OutputRelease { release }),
    }
}

fn retry_buffer_release(release: BufferRelease) -> BufferRelease {
    match release {
        BufferRelease::Pending(pending) => pending.retry(),
        release => release,
    }
}

fn release_error(release: &BufferRelease) -> Option<&(dyn std::error::Error + 'static)> {
    match release {
        BufferRelease::Released(_) => None,
        BufferRelease::Pending(pending) => Some(pending.error()),
        BufferRelease::Quarantined(quarantine) => Some(quarantine.error()),
    }
}
