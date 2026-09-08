//! # sched
//!
//! Deterministic CPU-only admission and residency coordination. It delegates
//! all validated identifiers and byte accounting to [`placement`], and never
//! initializes hardware or loads a model.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::BTreeMap;
use std::sync::Arc;

use placement::{PlacementRefusal, PlanRequest, PreparedPlan, ReservationLease, ReservationLedger};
use snafu::Snafu;

const INITIAL_IDENTIFIER: u64 = 1;
const DEFAULT_MAX_ACTIVE_ADMISSIONS: usize = 64;
const DEFAULT_MAX_PENDING_OPERATIONS: usize = 8;
const DEFAULT_MAX_ACTIVE_USES: usize = 32;
const MAX_RESIDENT_HANDLE_BYTES: usize = 256;

/// Bounded controller limits.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SchedulerLimits {
    admissions: usize,
    pending_operations: usize,
    uses: usize,
}

/// A controller-owned host resource-grant generation.
#[derive(Debug, Clone)]
pub struct GrantGeneration {
    brand: Arc<ControllerBrand>,
    value: u64,
}

/// An opaque prepared admission capability.
///
/// It is not a serialization type: v1 [`placement::PlanOutcome`] reports do
/// not authorize a runtime admission.
#[derive(Debug)]
pub struct PreparedAdmission {
    brand: Arc<ControllerBrand>,
    generation: u64,
    prepared: PreparedPlan,
}

/// An opaque per-workload committed admission capability.
#[derive(Debug, Clone)]
pub struct AdmissionTicket {
    brand: Arc<ControllerBrand>,
    generation: u64,
    admission_id: u64,
}

/// An opaque identity for one scheduler-issued executor operation.
#[derive(Debug, Clone)]
pub struct OperationId {
    brand: Arc<ControllerBrand>,
    value: u64,
}

/// An opaque capability for one active resident use.
#[derive(Debug)]
pub struct UsePermit {
    brand: Arc<ControllerBrand>,
    value: u64,
}

/// A bounded executor-local identity for a successfully loaded resident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentHandle(String);

/// A single physical action that a future trusted native executor may perform.
#[derive(Debug, Clone)]
pub struct RuntimeCommand {
    operation: OperationId,
    kind: RuntimeCommandKind,
}

/// One bounded result from [`Scheduler::poll_command`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PollOutcome {
    /// One physical executor command is ready.
    Command(RuntimeCommand),
    /// One local release completed without issuing a physical command.
    Progressed,
    /// A command is ready but the pending-operation bound is full.
    PendingLimit,
    /// No release or command can make progress until a caller supplies input.
    Idle,
}

/// Non-authoritative data supplied with a runtime command.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeCommandKind {
    /// Load the validated placement named by these checked facts.
    Load {
        /// Profile identifier.
        profile_id: String,
        /// Immutable artifact identity.
        artifact_id: String,
        /// Immutable artifact digest.
        digest: String,
        /// Declared device identity.
        device_id: String,
        /// Checked planner estimate retained by this reservation.
        total_estimated_bytes: u64,
    },
    /// Evict the trusted executor-local resident.
    Evict {
        /// Executor-local resident identity.
        resident: ResidentHandle,
    },
}

/// A trusted executor acknowledgement for an issued command.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RuntimeCompletion {
    /// A load completed and produced this resident identity.
    Loaded {
        /// Issued load operation.
        operation: OperationId,
        /// Trusted executor-local resident identity.
        resident: ResidentHandle,
    },
    /// The executor confirms the failed load retained no allocation.
    LoadFailed {
        /// Issued load operation.
        operation: OperationId,
    },
    /// The executor confirms that eviction reclaimed the allocation.
    Evicted {
        /// Issued eviction operation.
        operation: OperationId,
    },
    /// The executor failed to evict; the allocation remains retained.
    EvictFailed {
        /// Issued eviction operation.
        operation: OperationId,
    },
}

/// Typed scheduler refusal or transition error.
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum SchedulerError {
    /// Placement's accounting authority refused the operation.
    #[snafu(display("placement accounting refused the operation: {source}"))]
    Placement {
        /// Placement refusal.
        source: PlacementRefusal,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A bound was zero.
    #[snafu(display("scheduler limit {field} must be greater than zero"))]
    InvalidLimit {
        /// Invalid limit name.
        field: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A capability was issued by another controller instance.
    #[snafu(display("{kind} belongs to another scheduler"))]
    ForeignCapability {
        /// Rejected capability kind.
        kind: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A capability was issued by a no-longer-current grant generation.
    #[snafu(display("resource grant generation is stale"))]
    StaleGeneration {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// The controller's current grant is revoked.
    #[snafu(display("resource grant is revoked"))]
    GrantRevoked {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A grant must be revoked before replacement.
    #[snafu(display("resource grant must be revoked before replacement"))]
    GrantNotRevoked {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A revoked grant still retains leases, commands, or uses.
    #[snafu(display("resource grant has not fully drained"))]
    GrantNotDrained {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A batch would exceed the active admission bound.
    #[snafu(display("active admission limit would be exceeded"))]
    ActiveAdmissionLimit {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A new use would exceed the permit bound.
    #[snafu(display("active use limit would be exceeded"))]
    ActiveUseLimit {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A monotonic controller identifier would wrap.
    #[snafu(display("scheduler {kind} identifier overflowed"))]
    IdentifierOverflow {
        /// Identifier category.
        kind: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A ticket no longer refers to a retained admission.
    #[snafu(display("admission ticket is unknown or released"))]
    UnknownAdmission {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// An admission is not resident and eligible for a new use.
    #[snafu(display("admission is not resident and available for use"))]
    AdmissionNotResident {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// An operation is duplicate, late, or unknown.
    #[snafu(display("operation is unknown or already acknowledged"))]
    UnknownOperation {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A completion type does not match its issued operation.
    #[snafu(display("completion kind does not match the issued operation"))]
    OperationKindMismatch {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A completion does not match current admission state.
    #[snafu(display("completion does not match the admission state"))]
    OperationStateMismatch {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A use permit is duplicate, late, or unknown.
    #[snafu(display("use permit is unknown or already finished"))]
    UnknownUsePermit {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A resident handle is empty or exceeds the protocol bound.
    #[snafu(display("resident handle must be nonempty and at most {max} bytes"))]
    InvalidResidentHandle {
        /// Handle byte bound.
        max: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A load completion reused a handle held by another live admission.
    #[snafu(display("resident handle is already live in another admission"))]
    DuplicateResidentHandle {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Single-owner deterministic admission and residency controller.
///
/// One controller is process-local accounting, not an exclusive host claim.
/// The future service owner must instantiate exactly one controller per granted
/// resource set and drain or reconcile it during shutdown and restart.
#[derive(Debug)]
pub struct Scheduler {
    brand: Arc<ControllerBrand>,
    ledger: ReservationLedger,
    generation: u64,
    revoked: bool,
    limits: SchedulerLimits,
    admissions: BTreeMap<u64, Admission>,
    operations: BTreeMap<u64, PendingOperation>,
    permits: BTreeMap<u64, u64>,
    next_admission_id: u64,
    next_operation_id: u64,
    next_permit_id: u64,
    next_drain_cursor: u64,
}

#[derive(Debug)]
struct ControllerBrand;

#[derive(Debug)]
struct Admission {
    lease: ReservationLease,
    state: AdmissionState,
    active_uses: usize,
}

#[derive(Debug)]
enum AdmissionState {
    Reserved,
    Loading,
    Resident(ResidentHandle),
    InUse(ResidentHandle),
    Draining {
        resident: Option<ResidentHandle>,
        loading: bool,
    },
    Evicting(ResidentHandle),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationKind {
    Load,
    Evict,
}

#[derive(Debug)]
struct PendingOperation {
    admission_id: u64,
    kind: OperationKind,
}

impl Default for SchedulerLimits {
    fn default() -> Self {
        Self {
            admissions: DEFAULT_MAX_ACTIVE_ADMISSIONS,
            pending_operations: DEFAULT_MAX_PENDING_OPERATIONS,
            uses: DEFAULT_MAX_ACTIVE_USES,
        }
    }
}

impl SchedulerLimits {
    /// Construct explicit nonzero bounds.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::InvalidLimit`] when any bound is zero.
    pub fn try_new(
        max_active_admissions: usize,
        max_pending_operations: usize,
        max_active_uses: usize,
    ) -> Result<Self, SchedulerError> {
        for (field, value) in [
            ("max_active_admissions", max_active_admissions),
            ("max_pending_operations", max_pending_operations),
            ("max_active_uses", max_active_uses),
        ] {
            if value == 0 {
                return Err(SchedulerError::InvalidLimit {
                    field,
                    location: error_location(),
                });
            }
        }
        Ok(Self {
            admissions: max_active_admissions,
            pending_operations: max_pending_operations,
            uses: max_active_uses,
        })
    }
}

impl ResidentHandle {
    /// Create a bounded executor-local resident identity.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::InvalidResidentHandle`] for empty or oversized
    /// values. The scheduler never interprets a valid value.
    pub fn try_new(value: impl Into<String>) -> Result<Self, SchedulerError> {
        let value = value.into();
        if value.is_empty() || value.len() > MAX_RESIDENT_HANDLE_BYTES {
            return Err(SchedulerError::InvalidResidentHandle {
                max: MAX_RESIDENT_HANDLE_BYTES,
                location: error_location(),
            });
        }
        Ok(Self(value))
    }

    /// Borrow this opaque executor-local value.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl RuntimeCommand {
    /// Return the identity that must be acknowledged exactly once.
    #[must_use]
    pub fn operation(&self) -> OperationId {
        self.operation.clone()
    }

    /// Borrow the non-authoritative executor instruction.
    #[must_use]
    pub fn kind(&self) -> &RuntimeCommandKind {
        &self.kind
    }
}

impl Scheduler {
    /// Create a controller bound to one validated immutable resource grant.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::Placement`] when the grant's static capacity
    /// and commitments cannot be represented by placement accounting.
    pub fn new(grant: &PlanRequest, limits: SchedulerLimits) -> Result<Self, SchedulerError> {
        let ledger = ReservationLedger::new(grant).map_err(placement_error)?;
        Ok(Self {
            brand: Arc::new(ControllerBrand),
            ledger,
            generation: INITIAL_IDENTIFIER,
            revoked: false,
            limits,
            admissions: BTreeMap::new(),
            operations: BTreeMap::new(),
            permits: BTreeMap::new(),
            next_admission_id: INITIAL_IDENTIFIER,
            next_operation_id: INITIAL_IDENTIFIER,
            next_permit_id: INITIAL_IDENTIFIER,
            next_drain_cursor: INITIAL_IDENTIFIER,
        })
    }

    /// Return the controller-owned current grant generation.
    #[must_use]
    pub fn generation(&self) -> GrantGeneration {
        GrantGeneration {
            brand: Arc::clone(&self.brand),
            value: self.generation,
        }
    }

    /// Prepare requested work against current reservations without mutation.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal when the grant is revoked, resource facts differ,
    /// planner accounting refuses the work, or its batch exceeds a bound.
    pub fn prepare(&mut self, request: &PlanRequest) -> Result<PreparedAdmission, SchedulerError> {
        self.ensure_not_revoked()?;
        self.ensure_admission_capacity(request.workload_count())?;
        let prepared = self.ledger.prepare(request).map_err(placement_error)?;
        self.ensure_admission_capacity(prepared.placement_count())?;
        Ok(PreparedAdmission {
            brand: Arc::clone(&self.brand),
            generation: self.generation,
            prepared,
        })
    }

    /// Commit one prepared batch as distinct accounting leases.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal with no state change when authority, generation,
    /// limits, or placement revalidation fails.
    pub fn commit(
        &mut self,
        prepared: PreparedAdmission,
    ) -> Result<Vec<AdmissionTicket>, SchedulerError> {
        self.ensure_local(&prepared.brand, "prepared admission")?;
        self.ensure_current(prepared.generation)?;
        self.ensure_not_revoked()?;
        self.ensure_admission_capacity(prepared.prepared.placement_count())?;
        let count = u64::try_from(prepared.prepared.placement_count()).map_err(|_| {
            SchedulerError::IdentifierOverflow {
                kind: "admission",
                location: error_location(),
            }
        })?;
        let next_admission_id = self.next_admission_id.checked_add(count).ok_or(
            SchedulerError::IdentifierOverflow {
                kind: "admission",
                location: error_location(),
            },
        )?;
        let leases = self
            .ledger
            .commit(prepared.prepared)
            .map_err(placement_error)?;
        let mut tickets = Vec::with_capacity(leases.len());
        let mut admission_id = self.next_admission_id;
        for lease in leases {
            self.admissions.insert(
                admission_id,
                Admission {
                    lease,
                    state: AdmissionState::Reserved,
                    active_uses: 0,
                },
            );
            tickets.push(self.ticket(admission_id));
            admission_id =
                admission_id
                    .checked_add(1)
                    .ok_or(SchedulerError::IdentifierOverflow {
                        kind: "admission",
                        location: error_location(),
                    })?;
        }
        self.next_admission_id = next_admission_id;
        Ok(tickets)
    }

    /// Revoke the current grant immediately and begin safe draining.
    ///
    /// Repeated revocation is idempotent. It denies every new prepare, commit,
    /// and use before physical executor actions drain through [`Self::poll_command`].
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign or stale generation.
    pub fn revoke(&mut self, generation: &GrantGeneration) -> Result<(), SchedulerError> {
        self.ensure_local(&generation.brand, "grant generation")?;
        self.ensure_current(generation.value)?;
        if self.revoked {
            return Ok(());
        }
        self.revoked = true;
        for admission in self.admissions.values_mut() {
            Self::transition_to_draining(admission);
        }
        Ok(())
    }

    /// Request that one admission stop accepting new uses and release safely.
    ///
    /// The request denies new uses immediately. A reserved lease releases through
    /// [`Self::poll_command`]; a loaded resident releases only after its trusted
    /// eviction acknowledgement. Repeated requests for an active or already
    /// released local admission are idempotent. This does not revoke the grant or
    /// affect unrelated admissions.
    ///
    /// # Errors
    ///
    /// Returns an error without mutation for a foreign or stale ticket.
    pub fn request_retirement(&mut self, ticket: &AdmissionTicket) -> Result<(), SchedulerError> {
        self.ensure_local(&ticket.brand, "admission ticket")?;
        self.ensure_current(ticket.generation)?;
        let Some(admission) = self.admissions.get_mut(&ticket.admission_id) else {
            return Ok(());
        };
        Self::transition_to_draining(admission);
        Ok(())
    }

    /// Install a new resource grant after the revoked prior grant fully drains.
    ///
    /// # Errors
    ///
    /// Returns a typed error with no mutation unless the old grant is revoked,
    /// has no retained state, and the replacement passes placement validation.
    pub fn replace_grant(
        &mut self,
        grant: &PlanRequest,
    ) -> Result<GrantGeneration, SchedulerError> {
        if !self.revoked {
            return Err(SchedulerError::GrantNotRevoked {
                location: error_location(),
            });
        }
        if !self.admissions.is_empty() || !self.operations.is_empty() || !self.permits.is_empty() {
            return Err(SchedulerError::GrantNotDrained {
                location: error_location(),
            });
        }
        let generation =
            self.generation
                .checked_add(1)
                .ok_or(SchedulerError::IdentifierOverflow {
                    kind: "generation",
                    location: error_location(),
                })?;
        let ledger = ReservationLedger::new(grant).map_err(placement_error)?;
        self.ledger = ledger;
        self.generation = generation;
        self.revoked = false;
        Ok(self.generation())
    }

    /// Poll one deterministic command, subject to the pending-operation bound.
    ///
    /// Callers immediately poll again after [`PollOutcome::Progressed`]. Only
    /// [`PollOutcome::Idle`] is quiescent; [`PollOutcome::PendingLimit`] needs
    /// an executor acknowledgement before another physical command can issue.
    /// Revocation never calls this method, so a full pending bound cannot delay
    /// revocation; acknowledgements make room for later drain commands.
    ///
    /// # Errors
    ///
    /// Returns an error without mutation if the operation ID would wrap.
    pub fn poll_command(&mut self) -> Result<PollOutcome, SchedulerError> {
        if let Some(admission_id) = self.next_local_release() {
            self.release_admission(admission_id)?;
            return Ok(PollOutcome::Progressed);
        }
        if self.operations.len() >= self.limits.pending_operations {
            return Ok(PollOutcome::PendingLimit);
        }
        let Some((admission_id, operation_kind)) = self.next_command() else {
            return Ok(PollOutcome::Idle);
        };
        let operation_id = self.next_operation_id;
        let next_operation_id =
            operation_id
                .checked_add(1)
                .ok_or(SchedulerError::IdentifierOverflow {
                    kind: "operation",
                    location: error_location(),
                })?;
        let admission =
            self.admissions
                .get_mut(&admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        let kind = match operation_kind {
            OperationKind::Load => {
                admission.state = AdmissionState::Loading;
                RuntimeCommandKind::Load {
                    profile_id: admission.lease.profile_id().to_owned(),
                    artifact_id: admission.lease.artifact_id().to_owned(),
                    digest: admission.lease.digest().to_owned(),
                    device_id: admission.lease.device_id().to_owned(),
                    total_estimated_bytes: admission.lease.total_estimated_bytes(),
                }
            }
            OperationKind::Evict => {
                let AdmissionState::Draining {
                    resident: Some(resident),
                    loading: false,
                } = &admission.state
                else {
                    return Err(SchedulerError::OperationStateMismatch {
                        location: error_location(),
                    });
                };
                let resident = resident.clone();
                admission.state = AdmissionState::Evicting(resident.clone());
                RuntimeCommandKind::Evict { resident }
            }
        };
        self.operations.insert(
            operation_id,
            PendingOperation {
                admission_id,
                kind: operation_kind,
            },
        );
        self.next_operation_id = next_operation_id;
        Ok(PollOutcome::Command(RuntimeCommand {
            operation: OperationId {
                brand: Arc::clone(&self.brand),
                value: operation_id,
            },
            kind,
        }))
    }

    /// Apply one trusted executor acknowledgement.
    ///
    /// A load failure releases only because this completion explicitly attests
    /// that no allocation remains. Eviction failure retains the lease.
    ///
    /// # Errors
    ///
    /// Returns a typed error without mutation for foreign, duplicate, late, or
    /// state/kind-mismatched acknowledgements.
    pub fn complete(&mut self, completion: RuntimeCompletion) -> Result<(), SchedulerError> {
        let operation = completion.operation();
        self.ensure_local(&operation.brand, "operation")?;
        let Some(pending) = self.operations.get(&operation.value) else {
            return Err(SchedulerError::UnknownOperation {
                location: error_location(),
            });
        };
        if pending.kind != completion.kind() {
            return Err(SchedulerError::OperationKindMismatch {
                location: error_location(),
            });
        }
        let admission_id = pending.admission_id;
        self.ensure_completion_state(admission_id, pending.kind)?;
        match completion {
            RuntimeCompletion::Loaded { resident, .. } => {
                if self.resident_is_live(&resident) {
                    return Err(SchedulerError::DuplicateResidentHandle {
                        location: error_location(),
                    });
                }
                self.operations.remove(&operation.value);
                let admission = self.admissions.get_mut(&admission_id).ok_or(
                    SchedulerError::UnknownAdmission {
                        location: error_location(),
                    },
                )?;
                let drain_after_load = self.revoked
                    || matches!(
                        &admission.state,
                        AdmissionState::Draining {
                            resident: None,
                            loading: true,
                        }
                    );
                admission.state = if drain_after_load {
                    AdmissionState::Draining {
                        resident: Some(resident),
                        loading: false,
                    }
                } else {
                    AdmissionState::Resident(resident)
                };
            }
            RuntimeCompletion::LoadFailed { .. } | RuntimeCompletion::Evicted { .. } => {
                self.release_admission(admission_id)?;
                self.operations.remove(&operation.value);
            }
            RuntimeCompletion::EvictFailed { .. } => {
                self.operations.remove(&operation.value);
                let admission = self.admissions.get_mut(&admission_id).ok_or(
                    SchedulerError::UnknownAdmission {
                        location: error_location(),
                    },
                )?;
                let AdmissionState::Evicting(resident) = &admission.state else {
                    return Err(SchedulerError::OperationStateMismatch {
                        location: error_location(),
                    });
                };
                admission.state = AdmissionState::Draining {
                    resident: Some(resident.clone()),
                    loading: false,
                };
                self.next_drain_cursor = admission_id.checked_add(1).unwrap_or(INITIAL_IDENTIFIER);
            }
        }
        Ok(())
    }

    /// Start a bounded active use of a resident admission.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign, stale, revoked,
    /// nonresident, or over-limit tickets.
    pub fn begin_use(&mut self, ticket: &AdmissionTicket) -> Result<UsePermit, SchedulerError> {
        self.ensure_local(&ticket.brand, "admission ticket")?;
        self.ensure_current(ticket.generation)?;
        self.ensure_not_revoked()?;
        if self.permits.len() >= self.limits.uses {
            return Err(SchedulerError::ActiveUseLimit {
                location: error_location(),
            });
        }
        let permit_id = self.next_permit_id;
        let next_permit_id =
            permit_id
                .checked_add(1)
                .ok_or(SchedulerError::IdentifierOverflow {
                    kind: "use permit",
                    location: error_location(),
                })?;
        let admission = self.admissions.get_mut(&ticket.admission_id).ok_or(
            SchedulerError::UnknownAdmission {
                location: error_location(),
            },
        )?;
        let resident = match &admission.state {
            AdmissionState::Resident(resident) | AdmissionState::InUse(resident) => {
                resident.clone()
            }
            _ => {
                return Err(SchedulerError::AdmissionNotResident {
                    location: error_location(),
                });
            }
        };
        admission.active_uses =
            admission
                .active_uses
                .checked_add(1)
                .ok_or(SchedulerError::IdentifierOverflow {
                    kind: "active use",
                    location: error_location(),
                })?;
        admission.state = AdmissionState::InUse(resident);
        self.permits.insert(permit_id, ticket.admission_id);
        self.next_permit_id = next_permit_id;
        Ok(UsePermit {
            brand: Arc::clone(&self.brand),
            value: permit_id,
        })
    }

    /// Finish one active use exactly once.
    ///
    /// The borrowed permit remains retryable after a foreign-controller or
    /// other rejected attempt; a successful finish removes its live identity.
    ///
    /// # Errors
    ///
    /// Returns a typed error without mutation for foreign or duplicate permits.
    pub fn finish_use(&mut self, permit: &UsePermit) -> Result<(), SchedulerError> {
        self.ensure_local(&permit.brand, "use permit")?;
        let Some(&admission_id) = self.permits.get(&permit.value) else {
            return Err(SchedulerError::UnknownUsePermit {
                location: error_location(),
            });
        };
        let admission =
            self.admissions
                .get_mut(&admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        let next_uses =
            admission
                .active_uses
                .checked_sub(1)
                .ok_or(SchedulerError::UnknownUsePermit {
                    location: error_location(),
                })?;
        let state = match &admission.state {
            AdmissionState::InUse(resident) if next_uses == 0 && self.revoked => {
                AdmissionState::Draining {
                    resident: Some(resident.clone()),
                    loading: false,
                }
            }
            AdmissionState::InUse(resident) if next_uses == 0 => {
                AdmissionState::Resident(resident.clone())
            }
            AdmissionState::InUse(resident) => AdmissionState::InUse(resident.clone()),
            AdmissionState::Draining {
                resident: Some(resident),
                loading: false,
            } if next_uses == 0 => AdmissionState::Draining {
                resident: Some(resident.clone()),
                loading: false,
            },
            AdmissionState::Draining {
                resident: Some(resident),
                loading: false,
            } => AdmissionState::Draining {
                resident: Some(resident.clone()),
                loading: false,
            },
            _ => {
                return Err(SchedulerError::AdmissionNotResident {
                    location: error_location(),
                });
            }
        };
        admission.active_uses = next_uses;
        admission.state = state;
        self.permits.remove(&permit.value);
        Ok(())
    }

    fn ensure_admission_capacity(&self, incoming: usize) -> Result<(), SchedulerError> {
        let total = self.admissions.len().checked_add(incoming).ok_or(
            SchedulerError::ActiveAdmissionLimit {
                location: error_location(),
            },
        )?;
        if total > self.limits.admissions {
            return Err(SchedulerError::ActiveAdmissionLimit {
                location: error_location(),
            });
        }
        Ok(())
    }

    fn ensure_local(
        &self,
        brand: &Arc<ControllerBrand>,
        kind: &'static str,
    ) -> Result<(), SchedulerError> {
        if Arc::ptr_eq(&self.brand, brand) {
            Ok(())
        } else {
            Err(SchedulerError::ForeignCapability {
                kind,
                location: error_location(),
            })
        }
    }

    fn ensure_current(&self, generation: u64) -> Result<(), SchedulerError> {
        if generation == self.generation {
            Ok(())
        } else {
            Err(SchedulerError::StaleGeneration {
                location: error_location(),
            })
        }
    }

    fn ensure_not_revoked(&self) -> Result<(), SchedulerError> {
        if self.revoked {
            Err(SchedulerError::GrantRevoked {
                location: error_location(),
            })
        } else {
            Ok(())
        }
    }

    fn transition_to_draining(admission: &mut Admission) {
        let state = match &admission.state {
            AdmissionState::Reserved => AdmissionState::Draining {
                resident: None,
                loading: false,
            },
            AdmissionState::Loading => AdmissionState::Draining {
                resident: None,
                loading: true,
            },
            AdmissionState::Resident(resident) | AdmissionState::InUse(resident) => {
                AdmissionState::Draining {
                    resident: Some(resident.clone()),
                    loading: false,
                }
            }
            AdmissionState::Draining { .. } | AdmissionState::Evicting(_) => return,
        };
        admission.state = state;
    }

    fn next_command(&self) -> Option<(u64, OperationKind)> {
        self.admissions
            .iter()
            .find_map(|(&id, admission)| {
                matches!(
                    (&admission.state, self.revoked, admission.active_uses),
                    (AdmissionState::Reserved, false, 0)
                )
                .then_some((id, OperationKind::Load))
            })
            .or_else(|| self.next_eviction_from(self.next_drain_cursor))
            .or_else(|| self.next_eviction_from(INITIAL_IDENTIFIER))
    }

    fn next_eviction_from(&self, start: u64) -> Option<(u64, OperationKind)> {
        self.admissions.range(start..).find_map(|(&id, admission)| {
            matches!(
                (&admission.state, admission.active_uses),
                (
                    AdmissionState::Draining {
                        resident: Some(_),
                        loading: false,
                    },
                    0,
                )
            )
            .then_some((id, OperationKind::Evict))
        })
    }

    fn next_local_release(&self) -> Option<u64> {
        self.admissions.iter().find_map(|(&id, admission)| {
            matches!(
                admission.state,
                AdmissionState::Draining {
                    resident: None,
                    loading: false,
                }
            )
            .then_some(id)
        })
    }

    fn resident_is_live(&self, candidate: &ResidentHandle) -> bool {
        self.admissions.values().any(|admission| {
            matches!(
                &admission.state,
                AdmissionState::Resident(resident)
                    | AdmissionState::InUse(resident)
                    | AdmissionState::Draining {
                        resident: Some(resident),
                        ..
                    }
                    | AdmissionState::Evicting(resident)
                    if resident == candidate
            )
        })
    }

    fn ensure_completion_state(
        &self,
        admission_id: u64,
        kind: OperationKind,
    ) -> Result<(), SchedulerError> {
        let admission =
            self.admissions
                .get(&admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        match (kind, &admission.state) {
            (
                OperationKind::Load,
                AdmissionState::Loading
                | AdmissionState::Draining {
                    resident: None,
                    loading: true,
                },
            )
            | (OperationKind::Evict, AdmissionState::Evicting(_)) => Ok(()),
            _ => Err(SchedulerError::OperationStateMismatch {
                location: error_location(),
            }),
        }
    }

    fn release_admission(&mut self, admission_id: u64) -> Result<(), SchedulerError> {
        let admission =
            self.admissions
                .remove(&admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        let Admission {
            lease,
            state,
            active_uses,
        } = admission;
        match self.ledger.release(lease) {
            Ok(()) => Ok(()),
            Err(failure) => {
                let (reason, lease) = failure.into_parts();
                self.admissions.insert(
                    admission_id,
                    Admission {
                        lease,
                        state,
                        active_uses,
                    },
                );
                Err(placement_error(reason))
            }
        }
    }

    fn ticket(&self, admission_id: u64) -> AdmissionTicket {
        AdmissionTicket {
            brand: Arc::clone(&self.brand),
            generation: self.generation,
            admission_id,
        }
    }
}

impl RuntimeCompletion {
    fn operation(&self) -> OperationId {
        match self {
            Self::Loaded { operation, .. }
            | Self::LoadFailed { operation }
            | Self::Evicted { operation }
            | Self::EvictFailed { operation } => operation.clone(),
        }
    }

    fn kind(&self) -> OperationKind {
        match self {
            Self::Loaded { .. } | Self::LoadFailed { .. } => OperationKind::Load,
            Self::Evicted { .. } | Self::EvictFailed { .. } => OperationKind::Evict,
        }
    }
}

fn placement_error(source: PlacementRefusal) -> SchedulerError {
    SchedulerError::Placement {
        source,
        location: error_location(),
    }
}

#[track_caller]
fn error_location() -> snafu::Location {
    core::panic::Location::caller()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn request(workloads: &str, device_bytes: u64) -> Result<PlanRequest, SchedulerError> {
        let json = format!(
            r#"{{"schema_version":1,"devices":[{{"id":"w7900","gfx_isa":"gfx1100","total_bytes":{device_bytes},"reserved_bytes":0,"availability":"available"}}],"artifacts":[{{"artifact_id":"model","digest":"{DIGEST}"}}],"workloads":{workloads},"commitments":[]}}"#
        );
        PlanRequest::from_json(&json).map_err(placement_error)
    }

    fn workload(profile_id: &str, bytes: u64) -> String {
        format!(
            r#"{{"profile_id":"{profile_id}","artifact_id":"model","memory_estimate":{{"weights_bytes":{bytes},"kv_cache_bytes":0,"workspace_bytes":0,"headroom_bytes":0}},"placement":{{"kind":"requested_device","device_id":"w7900"}}}}"#
        )
    }

    fn one_ticket(
        scheduler: &mut Scheduler,
        request: &PlanRequest,
    ) -> Result<AdmissionTicket, SchedulerError> {
        let prepared = scheduler.prepare(request)?;
        let mut tickets = scheduler.commit(prepared)?;
        tickets.pop().ok_or(SchedulerError::UnknownAdmission {
            location: error_location(),
        })
    }

    fn poll_command(scheduler: &mut Scheduler) -> Result<RuntimeCommand, SchedulerError> {
        match scheduler.poll_command()? {
            PollOutcome::Command(command) => Ok(command),
            PollOutcome::Progressed | PollOutcome::PendingLimit | PollOutcome::Idle => {
                Err(SchedulerError::UnknownOperation {
                    location: error_location(),
                })
            }
        }
    }

    fn loaded_ticket(scheduler: &mut Scheduler) -> Result<AdmissionTicket, SchedulerError> {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let ticket = one_ticket(scheduler, &request)?;
        let command = poll_command(scheduler)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: command.operation(),
            resident: ResidentHandle::try_new("resident-main")?,
        })?;
        Ok(ticket)
    }

    #[test]
    fn cross_controller_and_stale_prepared_tokens_are_refused_without_mutation()
    -> Result<(), SchedulerError> {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut first = Scheduler::new(&request, SchedulerLimits::default())?;
        let mut second = Scheduler::new(&request, SchedulerLimits::default())?;
        let foreign = first.prepare(&request)?;
        assert!(
            matches!(
                second.commit(foreign),
                Err(SchedulerError::ForeignCapability { .. })
            ),
            "identical grant facts cannot authorize another controller"
        );
        assert!(
            second.admissions.is_empty(),
            "foreign commit must have no side effect"
        );

        let stale = first.prepare(&request)?;
        let current = first.prepare(&request)?;
        let _tickets = first.commit(current)?;
        assert!(
            matches!(
                first.commit(stale),
                Err(SchedulerError::Placement {
                    source: PlacementRefusal::StalePreparedPlan,
                    ..
                })
            ),
            "an intervening reservation must stale a prepared batch"
        );
        assert_eq!(first.admissions.len(), 1, "stale commit adds no admission");
        Ok(())
    }

    #[test]
    fn equal_numeric_foreign_capabilities_are_not_controller_authority()
    -> Result<(), SchedulerError> {
        let grant = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut first = Scheduler::new(&grant, SchedulerLimits::default())?;
        let mut second = Scheduler::new(&grant, SchedulerLimits::default())?;
        let first_ticket = loaded_ticket(&mut first)?;
        let second_ticket = loaded_ticket(&mut second)?;
        let _first_pending = one_ticket(&mut first, &grant)?;
        let first_operation = poll_command(&mut first)?.operation();
        let _second_pending = one_ticket(&mut second, &grant)?;
        let second_operation = poll_command(&mut second)?.operation();
        let first_permit = first.begin_use(&first_ticket)?;
        let second_permit = second.begin_use(&second_ticket)?;
        assert_eq!(
            first.generation().value,
            second.generation().value,
            "independent controllers begin with equal generation counters"
        );
        assert_eq!(
            first_ticket.admission_id, second_ticket.admission_id,
            "independent controllers issue equal first ticket counters"
        );
        assert_eq!(
            first_operation.value, second_operation.value,
            "independent controllers issue equal operation counters"
        );
        assert_eq!(
            first_permit.value, second_permit.value,
            "independent controllers issue equal permit counters"
        );
        assert!(
            matches!(
                first.revoke(&second.generation()),
                Err(SchedulerError::ForeignCapability { .. })
            ),
            "a matching generation number from another controller is foreign"
        );
        assert!(
            matches!(
                first.begin_use(&second_ticket),
                Err(SchedulerError::ForeignCapability { .. })
            ),
            "a matching ticket number from another controller is foreign"
        );
        assert!(
            matches!(
                first.complete(RuntimeCompletion::LoadFailed {
                    operation: second_operation
                }),
                Err(SchedulerError::ForeignCapability { .. })
            ),
            "a matching operation number from another controller is foreign"
        );
        assert!(
            matches!(
                first.finish_use(&second_permit),
                Err(SchedulerError::ForeignCapability { .. })
            ),
            "a matching permit number from another controller is foreign"
        );
        second.finish_use(&second_permit)?;
        first.finish_use(&first_permit)?;
        Ok(())
    }

    #[test]
    fn batch_reservation_is_all_or_none_and_per_placement() -> Result<(), SchedulerError> {
        let grant = request(
            &format!("[{},{}]", workload("first", 6), workload("second", 6)),
            20,
        )?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let prepared = scheduler.prepare(&grant)?;
        let tickets = scheduler.commit(prepared)?;
        assert_eq!(tickets.len(), 2, "every workload retains its own lease");
        let excess = request(&format!("[{}]", workload("third", 10)), 20)?;
        assert!(
            matches!(
                scheduler.prepare(&excess),
                Err(SchedulerError::Placement { .. })
            ),
            "existing reservations make an over-capacity batch refuse"
        );
        assert_eq!(
            scheduler.admissions.len(),
            2,
            "refusal retains prior leases unchanged"
        );
        Ok(())
    }

    #[test]
    fn revoke_while_loading_drains_and_never_allows_new_use() -> Result<(), SchedulerError> {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut scheduler = Scheduler::new(&request, SchedulerLimits::try_new(4, 1, 2)?)?;
        let ticket = one_ticket(&mut scheduler, &request)?;
        let load = poll_command(&mut scheduler)?;
        let generation = scheduler.generation();
        scheduler.revoke(&generation)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: load.operation(),
            resident: ResidentHandle::try_new("resident-main")?,
        })?;
        assert!(
            matches!(
                scheduler.begin_use(&ticket),
                Err(SchedulerError::GrantRevoked { .. })
            ),
            "revocation must reject a post-load use"
        );
        let evict = poll_command(&mut scheduler)?;
        assert!(
            matches!(evict.kind(), RuntimeCommandKind::Evict { .. }),
            "drain must evict"
        );
        scheduler.complete(RuntimeCompletion::Evicted {
            operation: evict.operation(),
        })?;
        assert!(
            scheduler.admissions.is_empty(),
            "successful eviction releases one lease"
        );
        Ok(())
    }

    #[test]
    fn revoke_drains_multiple_reserved_leases_without_pending_capacity()
    -> Result<(), SchedulerError> {
        let grant = request(
            &format!("[{},{}]", workload("first", 4), workload("second", 4)),
            20,
        )?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::try_new(4, 1, 2)?)?;
        let prepared = scheduler.prepare(&grant)?;
        let _tickets = scheduler.commit(prepared)?;
        scheduler.revoke(&scheduler.generation())?;
        assert_eq!(
            scheduler.admissions.len(),
            2,
            "revoke marks both leases draining before reclamation"
        );
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Progressed),
            "reserved leases report local progress without consuming a command slot"
        );
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Progressed),
            "each poll reports one bounded local reclamation step"
        );
        assert!(
            scheduler.admissions.is_empty(),
            "every reserved lease drains through the bounded poll path"
        );
        Ok(())
    }

    #[test]
    fn revoke_is_idempotent_and_blocks_commit_and_replacement_until_drain()
    -> Result<(), SchedulerError> {
        let grant = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let existing = scheduler.prepare(&grant)?;
        let _tickets = scheduler.commit(existing)?;
        let prepared = scheduler.prepare(&grant)?;
        let generation = scheduler.generation();
        scheduler.revoke(&generation)?;
        scheduler.revoke(&generation)?;
        assert!(
            matches!(
                scheduler.commit(prepared),
                Err(SchedulerError::GrantRevoked { .. })
            ),
            "revocation closes preprepared work before commit"
        );
        assert!(
            matches!(
                scheduler.replace_grant(&grant),
                Err(SchedulerError::GrantNotDrained { .. })
            ),
            "replacement waits for the revoked reservation to drain"
        );
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Progressed),
            "the revoked reserved lease advances through explicit local progress"
        );
        let replacement = scheduler.replace_grant(&grant)?;
        assert!(
            replacement.value > generation.value,
            "a drained replacement advances the controller generation"
        );
        Ok(())
    }

    #[test]
    fn in_use_admission_retains_lease_until_drain_and_evict_acknowledgement()
    -> Result<(), SchedulerError> {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut scheduler = Scheduler::new(&request, SchedulerLimits::default())?;
        let ticket = loaded_ticket(&mut scheduler)?;
        let first_permit = scheduler.begin_use(&ticket)?;
        let second_permit = scheduler.begin_use(&ticket)?;
        scheduler.revoke(&scheduler.generation())?;
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Idle),
            "active use prevents eviction until its permit is finished"
        );
        scheduler.finish_use(&first_permit)?;
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Idle),
            "one remaining use keeps a draining admission retained"
        );
        scheduler.finish_use(&second_permit)?;
        let evict = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::EvictFailed {
            operation: evict.operation(),
        })?;
        assert_eq!(
            scheduler.admissions.len(),
            1,
            "failed eviction keeps the accounting lease"
        );
        assert!(
            matches!(
                scheduler.complete(RuntimeCompletion::Evicted {
                    operation: evict.operation()
                }),
                Err(SchedulerError::UnknownOperation { .. })
            ),
            "late completion must not double-release the lease"
        );
        let retry = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Evicted {
            operation: retry.operation(),
        })?;
        assert!(
            scheduler.admissions.is_empty(),
            "successful retry releases the retained lease"
        );
        Ok(())
    }

    #[test]
    fn failed_eviction_rotates_to_another_draining_resident() -> Result<(), SchedulerError> {
        let grant = request(
            &format!("[{},{}]", workload("first", 4), workload("second", 4)),
            20,
        )?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::try_new(4, 1, 2)?)?;
        let prepared = scheduler.prepare(&grant)?;
        let _tickets = scheduler.commit(prepared)?;
        for resident in ["resident-first", "resident-second"] {
            let load = poll_command(&mut scheduler)?;
            scheduler.complete(RuntimeCompletion::Loaded {
                operation: load.operation(),
                resident: ResidentHandle::try_new(resident)?,
            })?;
        }
        scheduler.revoke(&scheduler.generation())?;
        let failed = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::EvictFailed {
            operation: failed.operation(),
        })?;
        let next = poll_command(&mut scheduler)?;
        assert!(
            matches!(
                next.kind(),
                RuntimeCommandKind::Evict { resident } if resident.as_str() == "resident-second"
            ),
            "a failed first eviction must not starve an independent draining resident"
        );
        Ok(())
    }

    #[test]
    fn load_completion_rejects_duplicate_live_handle_and_allows_reuse_after_eviction()
    -> Result<(), SchedulerError> {
        let grant = request(
            &format!("[{},{}]", workload("first", 4), workload("second", 4)),
            20,
        )?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let prepared = scheduler.prepare(&grant)?;
        let _tickets = scheduler.commit(prepared)?;
        let first_load = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: first_load.operation(),
            resident: ResidentHandle::try_new("shared-resident")?,
        })?;
        let second_load = poll_command(&mut scheduler)?;
        assert!(
            matches!(
                scheduler.complete(RuntimeCompletion::Loaded {
                    operation: second_load.operation(),
                    resident: ResidentHandle::try_new("shared-resident")?,
                }),
                Err(SchedulerError::DuplicateResidentHandle { .. })
            ),
            "one live executor handle cannot back two independent admissions"
        );
        assert_eq!(
            scheduler.operations.len(),
            1,
            "duplicate completion retains the pending load for retry"
        );
        scheduler.revoke(&scheduler.generation())?;
        let evict = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Evicted {
            operation: evict.operation(),
        })?;
        scheduler.complete(RuntimeCompletion::LoadFailed {
            operation: second_load.operation(),
        })?;
        let replacement = scheduler.replace_grant(&grant)?;
        assert!(
            replacement.value > 0,
            "successful eviction and load failure leave the grant replaceable"
        );
        let reuse_request = request(&format!("[{}]", workload("reuse", 4)), 20)?;
        let new_ticket = one_ticket(&mut scheduler, &reuse_request)?;
        let load = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: load.operation(),
            resident: ResidentHandle::try_new("shared-resident")?,
        })?;
        let permit = scheduler.begin_use(&new_ticket)?;
        scheduler.finish_use(&permit)?;
        Ok(())
    }

    #[test]
    fn local_reclamation_and_operation_overflow_are_separate_poll_outcomes()
    -> Result<(), SchedulerError> {
        let grant = request(
            &format!("[{},{}]", workload("first", 4), workload("second", 4)),
            20,
        )?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let prepared = scheduler.prepare(&grant)?;
        let _tickets = scheduler.commit(prepared)?;
        let load = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: load.operation(),
            resident: ResidentHandle::try_new("resident-first")?,
        })?;
        scheduler.revoke(&scheduler.generation())?;
        scheduler.next_operation_id = u64::MAX;
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Progressed),
            "one reserved lease drains successfully before any later command allocation"
        );
        let before = scheduler.admissions.len();
        assert!(
            matches!(
                scheduler.poll_command(),
                Err(SchedulerError::IdentifierOverflow { .. })
            ),
            "operation exhaustion is reported after local reclamation, not alongside it"
        );
        assert_eq!(
            scheduler.admissions.len(),
            before,
            "failed command allocation retains the independent resident lease"
        );
        Ok(())
    }

    #[test]
    fn rejected_transition_and_duplicate_permit_leave_state_unchanged() -> Result<(), SchedulerError>
    {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut scheduler = Scheduler::new(&request, SchedulerLimits::default())?;
        let ticket = one_ticket(&mut scheduler, &request)?;
        let before = (
            scheduler.admissions.len(),
            scheduler.operations.len(),
            scheduler.permits.len(),
        );
        assert!(
            matches!(
                scheduler.begin_use(&ticket),
                Err(SchedulerError::AdmissionNotResident { .. })
            ),
            "reserved work cannot be used before load acknowledgement"
        );
        assert_eq!(
            (
                scheduler.admissions.len(),
                scheduler.operations.len(),
                scheduler.permits.len()
            ),
            before,
            "rejected transition must be side-effect free"
        );
        let load = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: load.operation(),
            resident: ResidentHandle::try_new("resident-main")?,
        })?;
        let permit = scheduler.begin_use(&ticket)?;
        let duplicate = UsePermit {
            brand: Arc::clone(&permit.brand),
            value: permit.value,
        };
        scheduler.finish_use(&permit)?;
        assert!(
            matches!(
                scheduler.finish_use(&duplicate),
                Err(SchedulerError::UnknownUsePermit { .. })
            ),
            "duplicate permit cannot decrement use accounting twice"
        );
        Ok(())
    }

    #[test]
    fn retirement_of_reserved_admissions_is_idempotent_and_recovers_the_bound()
    -> Result<(), SchedulerError> {
        let initial = request(&format!("[{}]", workload("initial", 4)), 20)?;
        let mut scheduler = Scheduler::new(&initial, SchedulerLimits::try_new(1, 1, 1)?)?;
        for profile in ["first", "second", "third"] {
            let current = request(&format!("[{}]", workload(profile, 4)), 20)?;
            let ticket = one_ticket(&mut scheduler, &current)?;
            scheduler.request_retirement(&ticket)?;
            scheduler.request_retirement(&ticket)?;
            assert!(
                matches!(
                    scheduler.begin_use(&ticket),
                    Err(SchedulerError::AdmissionNotResident { .. })
                ),
                "retirement closes a reserved admission before a load can issue"
            );
            assert!(
                matches!(scheduler.poll_command()?, PollOutcome::Progressed),
                "a reserved admission releases without a physical command"
            );
            scheduler.request_retirement(&ticket)?;
            assert!(
                scheduler.admissions.is_empty(),
                "an already released local ticket remains an idempotent retirement request"
            );
        }
        Ok(())
    }

    #[test]
    fn retirement_cycles_reclaim_resident_admission_bound_without_grant_replacement()
    -> Result<(), SchedulerError> {
        let initial = request(&format!("[{}]", workload("initial", 4)), 20)?;
        let mut scheduler = Scheduler::new(&initial, SchedulerLimits::try_new(1, 1, 1)?)?;
        for profile in ["first", "second", "third"] {
            let current = request(&format!("[{}]", workload(profile, 4)), 20)?;
            let ticket = one_ticket(&mut scheduler, &current)?;
            let load = poll_command(&mut scheduler)?;
            scheduler.complete(RuntimeCompletion::Loaded {
                operation: load.operation(),
                resident: ResidentHandle::try_new(format!("resident-{profile}"))?,
            })?;
            let permit = scheduler.begin_use(&ticket)?;
            scheduler.finish_use(&permit)?;
            scheduler.request_retirement(&ticket)?;
            let evict = poll_command(&mut scheduler)?;
            scheduler.complete(RuntimeCompletion::Evicted {
                operation: evict.operation(),
            })?;
            assert!(
                scheduler.admissions.is_empty(),
                "every resident cycle returns its lease before the next admission"
            );
        }
        assert!(
            !scheduler.revoked,
            "selective retirement does not replace the grant"
        );
        Ok(())
    }

    #[test]
    fn retirement_while_loading_reclaims_a_late_success_only_after_eviction()
    -> Result<(), SchedulerError> {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut scheduler = Scheduler::new(&request, SchedulerLimits::try_new(2, 1, 2)?)?;
        let ticket = one_ticket(&mut scheduler, &request)?;
        let load = poll_command(&mut scheduler)?;
        scheduler.request_retirement(&ticket)?;
        scheduler.request_retirement(&ticket)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: load.operation(),
            resident: ResidentHandle::try_new("resident-main")?,
        })?;
        assert!(
            matches!(
                scheduler.begin_use(&ticket),
                Err(SchedulerError::AdmissionNotResident { .. })
            ),
            "a late successful load must remain retired under a live grant"
        );
        let evict = poll_command(&mut scheduler)?;
        scheduler.request_retirement(&ticket)?;
        scheduler.complete(RuntimeCompletion::Evicted {
            operation: evict.operation(),
        })?;
        scheduler.request_retirement(&ticket)?;
        assert!(
            scheduler.admissions.is_empty(),
            "only the eviction acknowledgement releases a late loaded resident"
        );
        Ok(())
    }

    #[test]
    fn retirement_while_loading_releases_only_after_a_failed_load_acknowledgement()
    -> Result<(), SchedulerError> {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut scheduler = Scheduler::new(&request, SchedulerLimits::default())?;
        let ticket = one_ticket(&mut scheduler, &request)?;
        let load = poll_command(&mut scheduler)?;
        scheduler.request_retirement(&ticket)?;
        scheduler.complete(RuntimeCompletion::LoadFailed {
            operation: load.operation(),
        })?;
        assert!(
            scheduler.admissions.is_empty(),
            "the trusted failed-load acknowledgement proves no allocation remains"
        );
        assert!(
            matches!(
                scheduler.complete(RuntimeCompletion::LoadFailed {
                    operation: load.operation()
                }),
                Err(SchedulerError::UnknownOperation { .. })
            ),
            "a duplicate late load acknowledgement cannot release accounting twice"
        );
        scheduler.request_retirement(&ticket)?;
        Ok(())
    }

    #[test]
    fn failed_retirement_eviction_retains_a_and_allows_b_load_and_use_progress()
    -> Result<(), SchedulerError> {
        let initial = request(&format!("[{}]", workload("alpha", 4)), 20)?;
        let mut scheduler = Scheduler::new(&initial, SchedulerLimits::try_new(3, 1, 2)?)?;
        let alpha = loaded_ticket(&mut scheduler)?;
        scheduler.request_retirement(&alpha)?;
        let failed_evict = poll_command(&mut scheduler)?;
        assert!(
            matches!(failed_evict.kind(), RuntimeCommandKind::Evict { .. }),
            "the retired resident issues its eviction before unrelated work is admitted"
        );
        scheduler.complete(RuntimeCompletion::EvictFailed {
            operation: failed_evict.operation(),
        })?;
        assert_eq!(
            scheduler.admissions.len(),
            1,
            "failed eviction keeps alpha's accounting lease"
        );
        let beta_request = request(&format!("[{}]", workload("beta", 4)), 20)?;
        let beta = one_ticket(&mut scheduler, &beta_request)?;
        let beta_load = poll_command(&mut scheduler)?;
        assert!(
            matches!(
                beta_load.kind(),
                RuntimeCommandKind::Load { profile_id, .. } if profile_id == "beta"
            ),
            "load-first dispatch lets unrelated reserved work progress after failed eviction"
        );
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: beta_load.operation(),
            resident: ResidentHandle::try_new("resident-beta")?,
        })?;
        let beta_use = scheduler.begin_use(&beta)?;
        scheduler.finish_use(&beta_use)?;
        let retry = poll_command(&mut scheduler)?;
        assert!(
            matches!(retry.kind(), RuntimeCommandKind::Evict { .. }),
            "the retained failed eviction becomes retryable after beta's load completes"
        );
        scheduler.complete(RuntimeCompletion::Evicted {
            operation: retry.operation(),
        })?;
        let later_beta_use = scheduler.begin_use(&beta)?;
        scheduler.finish_use(&later_beta_use)?;
        Ok(())
    }

    #[test]
    fn retirement_waits_for_live_uses_and_a_dropped_permit_is_not_an_acknowledgement()
    -> Result<(), SchedulerError> {
        let request = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut scheduler = Scheduler::new(&request, SchedulerLimits::default())?;
        let ticket = loaded_ticket(&mut scheduler)?;
        let first = scheduler.begin_use(&ticket)?;
        let second = scheduler.begin_use(&ticket)?;
        scheduler.request_retirement(&ticket)?;
        assert!(
            matches!(
                scheduler.begin_use(&ticket),
                Err(SchedulerError::AdmissionNotResident { .. })
            ),
            "retirement refuses every new use immediately"
        );
        scheduler.finish_use(&first)?;
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Idle),
            "one remaining live use retains the resident allocation"
        );
        scheduler.finish_use(&second)?;
        let evict = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Evicted {
            operation: evict.operation(),
        })?;

        let ticket = loaded_ticket(&mut scheduler)?;
        let abandoned = scheduler.begin_use(&ticket)?;
        scheduler.request_retirement(&ticket)?;
        drop(abandoned);
        assert!(
            matches!(scheduler.poll_command()?, PollOutcome::Idle),
            "dropping a caller-held permit cannot falsely acknowledge physical use completion"
        );
        assert_eq!(
            scheduler.admissions.len(),
            1,
            "the abandoned permit keeps its retired allocation accounted"
        );
        Ok(())
    }

    #[test]
    fn retirement_refuses_foreign_and_stale_tickets_without_mutation() -> Result<(), SchedulerError>
    {
        let grant = request(&format!("[{}]", workload("main", 4)), 20)?;
        let mut first = Scheduler::new(&grant, SchedulerLimits::default())?;
        let mut second = Scheduler::new(&grant, SchedulerLimits::default())?;
        let foreign = one_ticket(&mut second, &grant)?;
        assert!(
            matches!(
                first.request_retirement(&foreign),
                Err(SchedulerError::ForeignCapability { .. })
            ),
            "a ticket from an identical but distinct controller cannot retire work"
        );
        assert!(
            first.admissions.is_empty(),
            "foreign retirement has no side effect"
        );

        let ticket = one_ticket(&mut first, &grant)?;
        first.revoke(&first.generation())?;
        assert!(matches!(first.poll_command()?, PollOutcome::Progressed));
        let replacement = first.replace_grant(&grant)?;
        assert!(
            matches!(
                first.request_retirement(&ticket),
                Err(SchedulerError::StaleGeneration { .. })
            ),
            "a prior generation ticket cannot retire a new grant's admission"
        );
        assert!(
            replacement.value > 1,
            "replacement advanced the grant generation"
        );
        Ok(())
    }

    #[test]
    fn global_revoke_interleaves_with_selective_retirement_and_drains_all_admissions()
    -> Result<(), SchedulerError> {
        let grant = request(
            &format!("[{},{}]", workload("alpha", 4), workload("beta", 4)),
            20,
        )?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::try_new(3, 1, 2)?)?;
        let prepared = scheduler.prepare(&grant)?;
        let mut tickets = scheduler.commit(prepared)?;
        let beta = tickets.pop().ok_or(SchedulerError::UnknownAdmission {
            location: error_location(),
        })?;
        let alpha = tickets.pop().ok_or(SchedulerError::UnknownAdmission {
            location: error_location(),
        })?;
        for resident in ["resident-alpha", "resident-beta"] {
            let load = poll_command(&mut scheduler)?;
            scheduler.complete(RuntimeCompletion::Loaded {
                operation: load.operation(),
                resident: ResidentHandle::try_new(resident)?,
            })?;
        }
        scheduler.request_retirement(&alpha)?;
        let beta_use = scheduler.begin_use(&beta)?;
        scheduler.finish_use(&beta_use)?;
        scheduler.revoke(&scheduler.generation())?;
        assert!(
            matches!(
                scheduler.begin_use(&beta),
                Err(SchedulerError::GrantRevoked { .. })
            ),
            "global revocation remains the stronger all-admission authority"
        );
        for _ in 0..2 {
            let evict = poll_command(&mut scheduler)?;
            scheduler.complete(RuntimeCompletion::Evicted {
                operation: evict.operation(),
            })?;
        }
        assert!(
            scheduler.admissions.is_empty(),
            "selective and global drains release every lease exactly once"
        );
        Ok(())
    }

    #[test]
    fn bounded_sequence_preserves_drain_and_replace_invariants() -> Result<(), SchedulerError> {
        let initial = request(&format!("[{}]", workload("alpha", 4)), 20)?;
        let mut scheduler = Scheduler::new(&initial, SchedulerLimits::default())?;
        for profile in ["alpha", "beta", "gamma"] {
            let current = request(&format!("[{}]", workload(profile, 4)), 20)?;
            let ticket = one_ticket(&mut scheduler, &current)?;
            let load = poll_command(&mut scheduler)?;
            scheduler.complete(RuntimeCompletion::Loaded {
                operation: load.operation(),
                resident: ResidentHandle::try_new(format!("resident-{profile}"))?,
            })?;
            let permit = scheduler.begin_use(&ticket)?;
            scheduler.finish_use(&permit)?;
            scheduler.revoke(&scheduler.generation())?;
            let evict = poll_command(&mut scheduler)?;
            scheduler.complete(RuntimeCompletion::Evicted {
                operation: evict.operation(),
            })?;
            assert!(
                scheduler.admissions.is_empty(),
                "every sequence returns its only lease"
            );
            if profile != "gamma" {
                let replacement = request(&format!("[{}]", workload("next", 4)), 20)?;
                let _generation = scheduler.replace_grant(&replacement)?;
            }
        }
        Ok(())
    }
}
