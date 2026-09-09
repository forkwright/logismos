//! # sched
//!
//! Deterministic CPU-only admission and residency coordination. It delegates
//! all validated identifiers and byte accounting to [`placement`], and never
//! initializes hardware or loads a model.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use placement::{
    DeviceByteLease, DeviceByteReservationError, PlacementRefusal, PlanRequest, PreparedPlan,
    RequestedDeviceBytes, ReservationLease, ReservationLedger,
};
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

/// An opaque capability for one native resident load awaiting a trusted outcome.
#[derive(Debug)]
pub struct NativeLoadPermit {
    brand: Arc<ControllerBrand>,
    value: u64,
}

/// An opaque capability for one active native use with requested-byte custody.
#[derive(Debug)]
pub struct NativeUsePermit {
    brand: Arc<ControllerBrand>,
    value: u64,
}

/// An opaque capability retaining one completed native result's accounting.
#[derive(Debug)]
pub struct NativeResultLease {
    brand: Arc<ControllerBrand>,
    value: u64,
}

/// Requested host bytes retained by one native result.
///
/// This is a supplied accounting input, not a host grant, measured allocation,
/// allocator-overhead estimate, or evidence of physical capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestedHostBytes(NonZeroU64);

impl RequestedHostBytes {
    /// Construct one nonzero requested host-byte extent.
    #[must_use]
    pub const fn new(bytes: NonZeroU64) -> Self {
        Self(bytes)
    }

    /// Return the requested host-byte extent.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// A supplied controller-local ceiling for retained native host results.
///
/// This envelope is distinct from placement's declared device memory and is
/// not an authoritative host grant or a qualified physical-memory claim.
#[derive(Debug, Clone, Copy)]
pub struct NativeHostResultEnvelope {
    retained_bytes: u64,
}

impl NativeHostResultEnvelope {
    /// Construct an explicit retained-host-result accounting ceiling.
    #[must_use]
    pub const fn new(retained_bytes: u64) -> Self {
        Self { retained_bytes }
    }
}

/// Requested accounting for one native resident before its trusted load outcome.
///
/// These owner-derived requested bytes are not observed residency, allocator
/// overhead, or physical admission evidence.
#[derive(Debug, Clone)]
pub struct NativeResidentRequest {
    device_id: String,
    resident_bytes: RequestedDeviceBytes,
}

impl NativeResidentRequest {
    /// Bind one resident requested-byte charge to one declared device identity.
    #[must_use]
    pub fn new(device_id: impl Into<String>, resident_bytes: RequestedDeviceBytes) -> Self {
        Self {
            device_id: device_id.into(),
            resident_bytes,
        }
    }
}

/// Requested accounting for one native use and its optional retained output.
///
/// Mutable bytes cover only the use-local peak. An optional device output and
/// optional host output remain charged after trusted teardown until the result
/// is explicitly discarded.
#[derive(Debug, Clone, Copy)]
pub struct NativeUseRequest {
    mutable_device_bytes: RequestedDeviceBytes,
    retained_device_bytes: Option<RequestedDeviceBytes>,
    retained_host_bytes: Option<RequestedHostBytes>,
}

impl NativeUseRequest {
    /// Construct requested mutable and retained-result accounting for one use.
    #[must_use]
    pub const fn new(
        mutable_device_bytes: RequestedDeviceBytes,
        retained_device_bytes: Option<RequestedDeviceBytes>,
        retained_host_bytes: Option<RequestedHostBytes>,
    ) -> Self {
        Self {
            mutable_device_bytes,
            retained_device_bytes,
            retained_host_bytes,
        }
    }
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
    /// The executor cannot attest that a failed load released every allocated resource.
    ///
    /// This conservatively retains the admission lease without inventing a
    /// resident handle or permitting another load.
    LoadQuarantined {
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
    /// The executor cannot attest that an eviction left an intact resident or reclaimed it.
    ///
    /// This conservatively retains the admission lease and disables automatic
    /// eviction retry until a future trusted reconciliation protocol exists.
    EvictionQuarantined {
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
    /// Requested device-byte accounting refused a native operation.
    #[snafu(display("requested device-byte accounting refused the native operation: {source}"))]
    NativeDeviceBytes {
        /// Requested-byte accounting refusal.
        source: DeviceByteReservationError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A retained native host result would exceed its supplied accounting envelope.
    #[snafu(display(
        "retained native host result is exhausted: needs {required_bytes}, has {available_bytes}"
    ))]
    NativeHostResultExhausted {
        /// Requested retained host bytes.
        required_bytes: u64,
        /// Available bytes in the supplied accounting envelope.
        available_bytes: u64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Checked retained-native-host accounting arithmetic overflowed.
    #[snafu(display("retained native host result arithmetic overflowed"))]
    NativeHostResultArithmeticOverflow {
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
    /// A new native load would exceed the pending-operation bound.
    #[snafu(display("pending operation limit would be exceeded"))]
    PendingOperationLimit {
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
    native_loads: BTreeMap<u64, u64>,
    native_uses: BTreeMap<u64, NativeUseRecord>,
    native_quarantined_uses: BTreeMap<u64, NativeUseRecord>,
    native_results: BTreeMap<u64, NativeResultRecord>,
    host_result_envelope: NativeHostResultEnvelope,
    host_result_reserved: u64,
    next_admission_id: u64,
    next_operation_id: u64,
    next_permit_id: u64,
    next_native_load_id: u64,
    next_native_use_id: u64,
    next_native_result_id: u64,
    next_drain_cursor: u64,
}

#[derive(Debug)]
struct ControllerBrand;

#[derive(Debug)]
struct Admission {
    lease: AdmissionLease,
    state: AdmissionState,
    active_uses: usize,
}

#[derive(Debug)]
enum AdmissionLease {
    Legacy(ReservationLease),
    Native { resident: DeviceByteLease },
}

#[derive(Debug)]
struct NativeUseRecord {
    admission_id: u64,
    mutable_device: DeviceByteLease,
    retained_device: Option<DeviceByteLease>,
    retained_host: Option<HostResultLease>,
}

#[derive(Debug)]
struct NativeResultRecord {
    retained_device: Option<DeviceByteLease>,
    retained_host: Option<HostResultLease>,
}

#[derive(Debug)]
struct PreparedNativeUse {
    admission_id: u64,
    use_id: u64,
    next_use_id: u64,
    next_host_reserved: u64,
    device_id: String,
    next_uses: usize,
    resident: ResidentHandle,
}

#[derive(Debug)]
struct PreparedNativeFinish {
    admission_id: u64,
    next_uses: usize,
    next_state: AdmissionState,
    result_id: Option<u64>,
    next_result_id: u64,
}

#[derive(Debug)]
struct HostResultLease {
    requested: RequestedHostBytes,
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
    Quarantined {
        resident: Option<ResidentHandle>,
    },
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
        Self::new_with_native_host_results(grant, limits, NativeHostResultEnvelope::new(0))
    }

    /// Create a controller with an explicit supplied retained-host-result envelope.
    ///
    /// The envelope is independent of the declared device grant and supplies
    /// accounting only; it does not establish host capacity or physical grant
    /// authority.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::Placement`] when the grant's static capacity
    /// and commitments cannot be represented by placement accounting.
    pub fn new_with_native_host_results(
        grant: &PlanRequest,
        limits: SchedulerLimits,
        host_result_envelope: NativeHostResultEnvelope,
    ) -> Result<Self, SchedulerError> {
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
            native_loads: BTreeMap::new(),
            native_uses: BTreeMap::new(),
            native_quarantined_uses: BTreeMap::new(),
            native_results: BTreeMap::new(),
            host_result_envelope,
            host_result_reserved: 0,
            next_admission_id: INITIAL_IDENTIFIER,
            next_operation_id: INITIAL_IDENTIFIER,
            next_permit_id: INITIAL_IDENTIFIER,
            next_native_load_id: INITIAL_IDENTIFIER,
            next_native_use_id: INITIAL_IDENTIFIER,
            next_native_result_id: INITIAL_IDENTIFIER,
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
                    lease: AdmissionLease::Legacy(lease),
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

    /// Reserve one native resident charge and issue its trusted-load capability.
    ///
    /// This non-serialized path shares the controller's existing placement
    /// ledger but does not consume or add a v1 estimated placement lease.
    /// The returned load permit remains charged if dropped; the private trusted
    /// adapter must report a loaded, known-released, or quarantined outcome.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation when revoked, bounded, or when
    /// the requested resident bytes cannot reserve against the shared ledger.
    pub fn admit_native_resident(
        &mut self,
        request: NativeResidentRequest,
    ) -> Result<(AdmissionTicket, NativeLoadPermit), SchedulerError> {
        self.ensure_not_revoked()?;
        self.ensure_admission_capacity(1)?;
        self.ensure_pending_capacity()?;
        let admission_id = self.next_admission_id;
        let next_admission_id =
            admission_id
                .checked_add(1)
                .ok_or(SchedulerError::IdentifierOverflow {
                    kind: "admission",
                    location: error_location(),
                })?;
        let load_id = self.next_native_load_id;
        let next_load_id = load_id
            .checked_add(1)
            .ok_or(SchedulerError::IdentifierOverflow {
                kind: "native load permit",
                location: error_location(),
            })?;
        let resident = self
            .ledger
            .reserve_bytes(&request.device_id, request.resident_bytes)
            .map_err(native_device_error)?;
        self.admissions.insert(
            admission_id,
            Admission {
                lease: AdmissionLease::Native { resident },
                state: AdmissionState::Loading,
                active_uses: 0,
            },
        );
        self.native_loads.insert(load_id, admission_id);
        self.next_admission_id = next_admission_id;
        self.next_native_load_id = next_load_id;
        Ok((
            self.ticket(admission_id),
            NativeLoadPermit {
                brand: Arc::clone(&self.brand),
                value: load_id,
            },
        ))
    }

    /// Acknowledge that a native resident load completed with this handle.
    ///
    /// The private trusted adapter calls this only after its actual load
    /// completion. A duplicate handle leaves the load capability pending.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign, late, duplicate,
    /// or state-mismatched acknowledgement capabilities.
    pub fn complete_native_load(
        &mut self,
        permit: &NativeLoadPermit,
        resident: ResidentHandle,
    ) -> Result<(), SchedulerError> {
        self.ensure_local(&permit.brand, "native load permit")?;
        let admission_id =
            *self
                .native_loads
                .get(&permit.value)
                .ok_or(SchedulerError::UnknownOperation {
                    location: error_location(),
                })?;
        self.ensure_native_loading(admission_id)?;
        self.complete_load(admission_id, resident)?;
        self.native_loads.remove(&permit.value);
        Ok(())
    }

    /// Acknowledge a native load that retained no device allocation.
    ///
    /// The private trusted adapter calls this only after it has established the
    /// stated release outcome; dropping a load permit never releases bytes.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign, late, or
    /// state-mismatched acknowledgement capabilities.
    pub fn native_load_released(
        &mut self,
        permit: &NativeLoadPermit,
    ) -> Result<(), SchedulerError> {
        self.ensure_local(&permit.brand, "native load permit")?;
        let admission_id =
            *self
                .native_loads
                .get(&permit.value)
                .ok_or(SchedulerError::UnknownOperation {
                    location: error_location(),
                })?;
        self.ensure_native_loading(admission_id)?;
        self.release_admission(admission_id)?;
        self.native_loads.remove(&permit.value);
        Ok(())
    }

    /// Retain a native load's resident charge after an uncertain outcome.
    ///
    /// This terminal accounting state deliberately permits neither use nor
    /// automatic release. It is a trusted adapter report, not a physical proof.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign, late, or
    /// state-mismatched acknowledgement capabilities.
    pub fn quarantine_native_load(
        &mut self,
        permit: &NativeLoadPermit,
    ) -> Result<(), SchedulerError> {
        self.ensure_local(&permit.brand, "native load permit")?;
        let admission_id =
            *self
                .native_loads
                .get(&permit.value)
                .ok_or(SchedulerError::UnknownOperation {
                    location: error_location(),
                })?;
        self.ensure_native_loading(admission_id)?;
        self.quarantine_admission(admission_id)?;
        self.native_loads.remove(&permit.value);
        Ok(())
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
        if !self.admissions.is_empty()
            || !self.operations.is_empty()
            || !self.permits.is_empty()
            || !self.native_loads.is_empty()
            || !self.native_uses.is_empty()
            || !self.native_quarantined_uses.is_empty()
            || !self.native_results.is_empty()
            || self.host_result_reserved != 0
        {
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
        if self.pending_operation_count()? >= self.limits.pending_operations {
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
                let AdmissionLease::Legacy(lease) = &admission.lease else {
                    return Err(SchedulerError::OperationStateMismatch {
                        location: error_location(),
                    });
                };
                RuntimeCommandKind::Load {
                    profile_id: lease.profile_id().to_owned(),
                    artifact_id: lease.artifact_id().to_owned(),
                    digest: lease.digest().to_owned(),
                    device_id: lease.device_id().to_owned(),
                    total_estimated_bytes: lease.total_estimated_bytes(),
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
    /// that no allocation remains or every allocated leaf was logically
    /// released. `EvictFailed` is reserved for a known-intact resident;
    /// quarantined completions retain the full lease without retry.
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
                self.complete_load(admission_id, resident)?;
            }
            RuntimeCompletion::LoadFailed { .. } | RuntimeCompletion::Evicted { .. } => {
                self.release_admission(admission_id)?;
            }
            RuntimeCompletion::LoadQuarantined { .. }
            | RuntimeCompletion::EvictionQuarantined { .. } => {
                self.quarantine_admission(admission_id)?;
            }
            RuntimeCompletion::EvictFailed { .. } => {
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
        self.operations.remove(&operation.value);
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
        self.ensure_custody_capacity()?;
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
        if !matches!(&admission.lease, AdmissionLease::Legacy(_)) {
            return Err(SchedulerError::AdmissionNotResident {
                location: error_location(),
            });
        }
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

    /// Start one native use while reserving its mutable peak and retained output.
    ///
    /// The private adapter must retain this permit through actual native
    /// teardown. Dropping it never releases requested bytes or decrements use
    /// ownership; only [`Self::finish_native_use_after_teardown`] may do so.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign, stale, revoked,
    /// non-native, nonresident, over-limit, host-envelope, or shared-ledger
    /// accounting failures.
    pub fn begin_native_use(
        &mut self,
        ticket: &AdmissionTicket,
        request: NativeUseRequest,
    ) -> Result<NativeUsePermit, SchedulerError> {
        self.ensure_local(&ticket.brand, "admission ticket")?;
        self.ensure_current(ticket.generation)?;
        self.ensure_not_revoked()?;
        self.ensure_custody_capacity()?;
        let prepared = self.prepare_native_use(ticket, request)?;
        let (mutable_device, retained_device) = {
            let (admissions, ledger) = (&mut self.admissions, &mut self.ledger);
            let admission = admissions.get_mut(&prepared.admission_id).ok_or(
                SchedulerError::UnknownAdmission {
                    location: error_location(),
                },
            )?;
            let leases = ledger
                .reserve_bytes_batch(
                    &prepared.device_id,
                    request.mutable_device_bytes,
                    request
                        .retained_device_bytes
                        .map(|bytes| (prepared.device_id.as_str(), bytes)),
                )
                .map_err(native_device_error)?;
            admission.active_uses = prepared.next_uses;
            admission.state = AdmissionState::InUse(prepared.resident);
            leases
        };
        self.native_uses.insert(
            prepared.use_id,
            NativeUseRecord {
                admission_id: prepared.admission_id,
                mutable_device,
                retained_device,
                retained_host: request
                    .retained_host_bytes
                    .map(|requested| HostResultLease { requested }),
            },
        );
        self.host_result_reserved = prepared.next_host_reserved;
        self.next_native_use_id = prepared.next_use_id;
        Ok(NativeUsePermit {
            brand: Arc::clone(&self.brand),
            value: prepared.use_id,
        })
    }

    /// Finish one native use only after the private adapter acknowledges teardown.
    ///
    /// This releases the mutable requested-device charge and transfers any
    /// retained device or host output charge into an opaque result capability.
    /// It is not a cancellation acknowledgement and must not be called merely
    /// because a command was abandoned.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign, late, or
    /// state-mismatched permits. A device-release refusal restores the use and
    /// its still-live charge for retry or truthful quarantine.
    pub fn finish_native_use_after_teardown(
        &mut self,
        permit: &NativeUsePermit,
    ) -> Result<Option<NativeResultLease>, SchedulerError> {
        self.ensure_local(&permit.brand, "native use permit")?;
        let prepared = self.prepare_native_finish(permit.value)?;
        let brand = Arc::clone(&self.brand);
        let result = {
            let (admissions, native_uses, native_results, ledger) = (
                &mut self.admissions,
                &mut self.native_uses,
                &mut self.native_results,
                &mut self.ledger,
            );
            let admission = admissions.get_mut(&prepared.admission_id).ok_or(
                SchedulerError::UnknownAdmission {
                    location: error_location(),
                },
            )?;
            let mut record =
                native_uses
                    .remove(&permit.value)
                    .ok_or(SchedulerError::UnknownUsePermit {
                        location: error_location(),
                    })?;
            if let Err(failure) = ledger.release_bytes(record.mutable_device) {
                let (reason, lease) = failure.into_parts();
                record.mutable_device = lease;
                native_uses.insert(permit.value, record);
                return Err(native_device_error(reason));
            }
            admission.active_uses = prepared.next_uses;
            admission.state = prepared.next_state;
            prepared.result_id.map(|result_id| {
                native_results.insert(
                    result_id,
                    NativeResultRecord {
                        retained_device: record.retained_device,
                        retained_host: record.retained_host,
                    },
                );
                NativeResultLease {
                    brand,
                    value: result_id,
                }
            })
        };
        self.next_native_result_id = prepared.next_result_id;
        Ok(result)
    }

    /// Retain all native-use charges after a teardown outcome becomes uncertain.
    ///
    /// This terminal quarantine disables use and automatic eviction while the
    /// resident, mutable, and retained-output capabilities remain in custody.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign, late, or
    /// state-mismatched permits.
    pub fn quarantine_native_use(
        &mut self,
        permit: &NativeUsePermit,
    ) -> Result<(), SchedulerError> {
        self.ensure_local(&permit.brand, "native use permit")?;
        let record =
            self.native_uses
                .get(&permit.value)
                .ok_or(SchedulerError::UnknownUsePermit {
                    location: error_location(),
                })?;
        let resident =
            self.admissions
                .get(&record.admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        let resident = match &resident.state {
            AdmissionState::InUse(resident)
            | AdmissionState::Draining {
                resident: Some(resident),
                ..
            }
            | AdmissionState::Quarantined {
                resident: Some(resident),
            } => Some(resident.clone()),
            _ => {
                return Err(SchedulerError::OperationStateMismatch {
                    location: error_location(),
                });
            }
        };
        let record =
            self.native_uses
                .remove(&permit.value)
                .ok_or(SchedulerError::UnknownUsePermit {
                    location: error_location(),
                })?;
        let admission = self.admissions.get_mut(&record.admission_id).ok_or(
            SchedulerError::UnknownAdmission {
                location: error_location(),
            },
        )?;
        admission.state = AdmissionState::Quarantined { resident };
        self.native_quarantined_uses.insert(permit.value, record);
        Ok(())
    }

    /// Discard one retained native result after its private owner releases it.
    ///
    /// The result capability remains charged if device release cannot be
    /// acknowledged by the shared ledger; dropping it never releases bytes.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal without mutation for foreign or late result
    /// capabilities.
    pub fn discard_native_result(
        &mut self,
        result: &NativeResultLease,
    ) -> Result<(), SchedulerError> {
        self.ensure_local(&result.brand, "native result lease")?;
        let record =
            self.native_results
                .get(&result.value)
                .ok_or(SchedulerError::UnknownUsePermit {
                    location: error_location(),
                })?;
        let released_host = record
            .retained_host
            .as_ref()
            .map_or(0, |lease| lease.requested.get());
        let next_host_reserved = self.host_result_reserved.checked_sub(released_host).ok_or(
            SchedulerError::NativeHostResultArithmeticOverflow {
                location: error_location(),
            },
        )?;
        let record =
            self.native_results
                .remove(&result.value)
                .ok_or(SchedulerError::UnknownUsePermit {
                    location: error_location(),
                })?;
        if let Some(device) = record.retained_device {
            if let Err(failure) = self.ledger.release_bytes(device) {
                let (reason, device) = failure.into_parts();
                self.native_results.insert(
                    result.value,
                    NativeResultRecord {
                        retained_device: Some(device),
                        retained_host: record.retained_host,
                    },
                );
                return Err(native_device_error(reason));
            }
        }
        self.host_result_reserved = next_host_reserved;
        Ok(())
    }

    fn prepare_native_use(
        &self,
        ticket: &AdmissionTicket,
        request: NativeUseRequest,
    ) -> Result<PreparedNativeUse, SchedulerError> {
        let use_id = self.next_native_use_id;
        let next_use_id = use_id
            .checked_add(1)
            .ok_or(SchedulerError::IdentifierOverflow {
                kind: "native use permit",
                location: error_location(),
            })?;
        let next_host_reserved = self.next_host_result_reservation(request.retained_host_bytes)?;
        let admission =
            self.admissions
                .get(&ticket.admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        let AdmissionLease::Native { resident } = &admission.lease else {
            return Err(SchedulerError::AdmissionNotResident {
                location: error_location(),
            });
        };
        let resident = native_active_resident(&admission.state)?;
        let next_uses =
            admission
                .active_uses
                .checked_add(1)
                .ok_or(SchedulerError::IdentifierOverflow {
                    kind: "active native use",
                    location: error_location(),
                })?;
        Ok(PreparedNativeUse {
            admission_id: ticket.admission_id,
            use_id,
            next_use_id,
            next_host_reserved,
            device_id: resident.device_id().to_owned(),
            next_uses,
            resident,
        })
    }

    fn next_host_result_reservation(
        &self,
        retained_host: Option<RequestedHostBytes>,
    ) -> Result<u64, SchedulerError> {
        let requested_bytes = retained_host.map_or(0, RequestedHostBytes::get);
        let next_reserved = self
            .host_result_reserved
            .checked_add(requested_bytes)
            .ok_or(SchedulerError::NativeHostResultArithmeticOverflow {
                location: error_location(),
            })?;
        if next_reserved > self.host_result_envelope.retained_bytes {
            return Err(SchedulerError::NativeHostResultExhausted {
                required_bytes: requested_bytes,
                available_bytes: self
                    .host_result_envelope
                    .retained_bytes
                    .saturating_sub(self.host_result_reserved),
                location: error_location(),
            });
        }
        Ok(next_reserved)
    }

    fn prepare_native_finish(
        &self,
        permit_id: u64,
    ) -> Result<PreparedNativeFinish, SchedulerError> {
        let record = self
            .native_uses
            .get(&permit_id)
            .ok_or(SchedulerError::UnknownUsePermit {
                location: error_location(),
            })?;
        let result_id = (record.retained_device.is_some() || record.retained_host.is_some())
            .then_some(self.next_native_result_id);
        let next_result_id = match result_id {
            Some(identifier) => {
                identifier
                    .checked_add(1)
                    .ok_or(SchedulerError::IdentifierOverflow {
                        kind: "native result lease",
                        location: error_location(),
                    })?
            }
            None => self.next_native_result_id,
        };
        let admission =
            self.admissions
                .get(&record.admission_id)
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
        Ok(PreparedNativeFinish {
            admission_id: record.admission_id,
            next_uses,
            next_state: native_finished_state(&admission.state, next_uses, self.revoked)?,
            result_id,
            next_result_id,
        })
    }

    fn ensure_native_loading(&self, admission_id: u64) -> Result<(), SchedulerError> {
        let admission =
            self.admissions
                .get(&admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        if matches!(
            &admission.state,
            AdmissionState::Loading
                | AdmissionState::Draining {
                    resident: None,
                    loading: true,
                }
        ) {
            Ok(())
        } else {
            Err(SchedulerError::OperationStateMismatch {
                location: error_location(),
            })
        }
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

    fn ensure_pending_capacity(&self) -> Result<(), SchedulerError> {
        if self.pending_operation_count()? >= self.limits.pending_operations {
            return Err(SchedulerError::PendingOperationLimit {
                location: error_location(),
            });
        }
        Ok(())
    }

    fn pending_operation_count(&self) -> Result<usize, SchedulerError> {
        self.operations
            .len()
            .checked_add(self.native_loads.len())
            .ok_or(SchedulerError::PendingOperationLimit {
                location: error_location(),
            })
    }

    fn ensure_custody_capacity(&self) -> Result<(), SchedulerError> {
        let slots = self
            .permits
            .len()
            .checked_add(self.native_uses.len())
            .and_then(|value| value.checked_add(self.native_quarantined_uses.len()))
            .and_then(|value| value.checked_add(self.native_results.len()))
            .ok_or(SchedulerError::ActiveUseLimit {
                location: error_location(),
            })?;
        if slots >= self.limits.uses {
            return Err(SchedulerError::ActiveUseLimit {
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
            AdmissionState::Draining { .. }
            | AdmissionState::Evicting(_)
            | AdmissionState::Quarantined { .. } => return,
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

    fn complete_load(
        &mut self,
        admission_id: u64,
        resident: ResidentHandle,
    ) -> Result<(), SchedulerError> {
        if self.resident_handle_in_custody(&resident) {
            return Err(SchedulerError::DuplicateResidentHandle {
                location: error_location(),
            });
        }
        let admission =
            self.admissions
                .get_mut(&admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
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
        Ok(())
    }

    fn quarantine_admission(&mut self, admission_id: u64) -> Result<(), SchedulerError> {
        let admission =
            self.admissions
                .get_mut(&admission_id)
                .ok_or(SchedulerError::UnknownAdmission {
                    location: error_location(),
                })?;
        let resident = match &admission.state {
            AdmissionState::Evicting(resident) => Some(resident.clone()),
            AdmissionState::Loading
            | AdmissionState::Draining {
                resident: None,
                loading: true,
            } => None,
            _ => {
                return Err(SchedulerError::OperationStateMismatch {
                    location: error_location(),
                });
            }
        };
        admission.state = AdmissionState::Quarantined { resident };
        Ok(())
    }

    fn resident_handle_in_custody(&self, candidate: &ResidentHandle) -> bool {
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
                    | AdmissionState::Quarantined {
                        resident: Some(resident),
                    }
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
        match lease {
            AdmissionLease::Legacy(lease) => match self.ledger.release(lease) {
                Ok(()) => Ok(()),
                Err(failure) => {
                    let (reason, lease) = failure.into_parts();
                    self.admissions.insert(
                        admission_id,
                        Admission {
                            lease: AdmissionLease::Legacy(lease),
                            state,
                            active_uses,
                        },
                    );
                    Err(placement_error(reason))
                }
            },
            AdmissionLease::Native { resident } => match self.ledger.release_bytes(resident) {
                Ok(()) => Ok(()),
                Err(failure) => {
                    let (reason, resident) = failure.into_parts();
                    self.admissions.insert(
                        admission_id,
                        Admission {
                            lease: AdmissionLease::Native { resident },
                            state,
                            active_uses,
                        },
                    );
                    Err(native_device_error(reason))
                }
            },
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

fn native_finished_state(
    state: &AdmissionState,
    next_uses: usize,
    revoked: bool,
) -> Result<AdmissionState, SchedulerError> {
    match state {
        AdmissionState::InUse(resident) if next_uses == 0 && revoked => {
            Ok(AdmissionState::Draining {
                resident: Some(resident.clone()),
                loading: false,
            })
        }
        AdmissionState::InUse(resident) if next_uses == 0 => {
            Ok(AdmissionState::Resident(resident.clone()))
        }
        AdmissionState::InUse(resident) => Ok(AdmissionState::InUse(resident.clone())),
        AdmissionState::Draining {
            resident: Some(resident),
            loading: false,
        } => Ok(AdmissionState::Draining {
            resident: Some(resident.clone()),
            loading: false,
        }),
        AdmissionState::Quarantined { resident } => Ok(AdmissionState::Quarantined {
            resident: resident.clone(),
        }),
        _ => Err(SchedulerError::AdmissionNotResident {
            location: error_location(),
        }),
    }
}

fn native_active_resident(state: &AdmissionState) -> Result<ResidentHandle, SchedulerError> {
    match state {
        AdmissionState::Resident(resident) | AdmissionState::InUse(resident) => {
            Ok(resident.clone())
        }
        _ => Err(SchedulerError::AdmissionNotResident {
            location: error_location(),
        }),
    }
}

fn native_device_error(source: DeviceByteReservationError) -> SchedulerError {
    SchedulerError::NativeDeviceBytes {
        source,
        location: error_location(),
    }
}

impl RuntimeCompletion {
    fn operation(&self) -> OperationId {
        match self {
            Self::Loaded { operation, .. }
            | Self::LoadFailed { operation }
            | Self::LoadQuarantined { operation }
            | Self::Evicted { operation }
            | Self::EvictFailed { operation }
            | Self::EvictionQuarantined { operation } => operation.clone(),
        }
    }

    fn kind(&self) -> OperationKind {
        match self {
            Self::Loaded { .. } | Self::LoadFailed { .. } | Self::LoadQuarantined { .. } => {
                OperationKind::Load
            }
            Self::Evicted { .. } | Self::EvictFailed { .. } | Self::EvictionQuarantined { .. } => {
                OperationKind::Evict
            }
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

    fn requested_device_bytes(bytes: u64) -> Result<RequestedDeviceBytes, SchedulerError> {
        let bytes = NonZeroU64::new(bytes).ok_or(SchedulerError::InvalidLimit {
            field: "requested_device_bytes",
            location: error_location(),
        })?;
        Ok(RequestedDeviceBytes::new(bytes))
    }

    fn requested_host_bytes(bytes: u64) -> Result<RequestedHostBytes, SchedulerError> {
        let bytes = NonZeroU64::new(bytes).ok_or(SchedulerError::InvalidLimit {
            field: "requested_host_bytes",
            location: error_location(),
        })?;
        Ok(RequestedHostBytes::new(bytes))
    }

    fn native_loaded(
        scheduler: &mut Scheduler,
        resident_bytes: u64,
        resident: &str,
    ) -> Result<AdmissionTicket, SchedulerError> {
        let (ticket, load) = scheduler.admit_native_resident(NativeResidentRequest::new(
            "w7900",
            requested_device_bytes(resident_bytes)?,
        ))?;
        scheduler.complete_native_load(&load, ResidentHandle::try_new(resident)?)?;
        Ok(ticket)
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
    fn quarantined_load_retains_capacity_and_rejects_untrusted_completions()
    -> Result<(), SchedulerError> {
        let grant = request(&format!("[{}]", workload("main", 4)), 4)?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let ticket = one_ticket(&mut scheduler, &grant)?;
        let load = poll_command(&mut scheduler)?;
        let mut foreign = Scheduler::new(&grant, SchedulerLimits::default())?;
        assert!(matches!(
            foreign.complete(RuntimeCompletion::LoadQuarantined {
                operation: load.operation(),
            }),
            Err(SchedulerError::ForeignCapability { .. })
        ));
        assert!(matches!(
            scheduler.complete(RuntimeCompletion::EvictionQuarantined {
                operation: load.operation(),
            }),
            Err(SchedulerError::OperationKindMismatch { .. })
        ));
        assert_eq!(scheduler.operations.len(), 1);

        scheduler.complete(RuntimeCompletion::LoadQuarantined {
            operation: load.operation(),
        })?;
        assert!(matches!(
            scheduler
                .admissions
                .get(&ticket.admission_id)
                .map(|admission| &admission.state),
            Some(AdmissionState::Quarantined { resident: None })
        ));
        assert!(matches!(
            scheduler.begin_use(&ticket),
            Err(SchedulerError::AdmissionNotResident { .. })
        ));
        assert!(matches!(
            scheduler.prepare(&grant),
            Err(SchedulerError::Placement {
                source: PlacementRefusal::CapacityExhausted { .. },
                ..
            })
        ));
        assert!(matches!(scheduler.poll_command()?, PollOutcome::Idle));
        scheduler.revoke(&scheduler.generation())?;
        assert!(matches!(scheduler.poll_command()?, PollOutcome::Idle));
        assert!(matches!(
            scheduler.replace_grant(&grant),
            Err(SchedulerError::GrantNotDrained { .. })
        ));
        assert!(matches!(
            scheduler.complete(RuntimeCompletion::LoadQuarantined {
                operation: load.operation(),
            }),
            Err(SchedulerError::UnknownOperation { .. })
        ));
        Ok(())
    }

    #[test]
    fn quarantined_eviction_is_terminal_without_trusted_reconciliation()
    -> Result<(), SchedulerError> {
        let grant = request(&format!("[{}]", workload("main", 4)), 4)?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let ticket = loaded_ticket(&mut scheduler)?;
        scheduler.request_retirement(&ticket)?;
        let eviction = poll_command(&mut scheduler)?;
        assert!(matches!(
            scheduler.complete(RuntimeCompletion::LoadQuarantined {
                operation: eviction.operation(),
            }),
            Err(SchedulerError::OperationKindMismatch { .. })
        ));
        scheduler.complete(RuntimeCompletion::EvictionQuarantined {
            operation: eviction.operation(),
        })?;
        assert!(matches!(
            scheduler
                .admissions
                .get(&ticket.admission_id)
                .map(|admission| &admission.state),
            Some(AdmissionState::Quarantined {
                resident: Some(resident),
            }) if resident.as_str() == "resident-main"
        ));
        assert!(matches!(
            scheduler.begin_use(&ticket),
            Err(SchedulerError::AdmissionNotResident { .. })
        ));
        scheduler.request_retirement(&ticket)?;
        assert!(matches!(scheduler.poll_command()?, PollOutcome::Idle));
        assert!(matches!(
            scheduler.prepare(&grant),
            Err(SchedulerError::Placement {
                source: PlacementRefusal::CapacityExhausted { .. },
                ..
            })
        ));
        scheduler.revoke(&scheduler.generation())?;
        assert!(matches!(scheduler.poll_command()?, PollOutcome::Idle));
        assert!(matches!(
            scheduler.replace_grant(&grant),
            Err(SchedulerError::GrantNotDrained { .. })
        ));
        assert!(matches!(
            scheduler.complete(RuntimeCompletion::EvictionQuarantined {
                operation: eviction.operation(),
            }),
            Err(SchedulerError::UnknownOperation { .. })
        ));
        Ok(())
    }

    #[test]
    fn quarantined_eviction_keeps_its_handle_in_custody() -> Result<(), SchedulerError> {
        let first_request = request(&format!("[{}]", workload("first", 4)), 8)?;
        let mut scheduler = Scheduler::new(&first_request, SchedulerLimits::default())?;
        let first = one_ticket(&mut scheduler, &first_request)?;
        let first_load = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: first_load.operation(),
            resident: ResidentHandle::try_new("resident-first")?,
        })?;
        scheduler.request_retirement(&first)?;
        let eviction = poll_command(&mut scheduler)?;
        scheduler.complete(RuntimeCompletion::EvictionQuarantined {
            operation: eviction.operation(),
        })?;

        let second_request = request(&format!("[{}]", workload("second", 4)), 8)?;
        let second = one_ticket(&mut scheduler, &second_request)?;
        let second_load = poll_command(&mut scheduler)?;
        assert!(matches!(
            second_load.kind(),
            RuntimeCommandKind::Load { profile_id, .. } if profile_id == "second"
        ));
        assert!(matches!(
            scheduler.complete(RuntimeCompletion::Loaded {
                operation: second_load.operation(),
                resident: ResidentHandle::try_new("resident-first")?,
            }),
            Err(SchedulerError::DuplicateResidentHandle { .. })
        ));
        assert_eq!(
            scheduler.operations.len(),
            1,
            "duplicate handle acknowledgement leaves its load pending"
        );
        scheduler.complete(RuntimeCompletion::Loaded {
            operation: second_load.operation(),
            resident: ResidentHandle::try_new("resident-second")?,
        })?;
        let permit = scheduler.begin_use(&second)?;
        scheduler.finish_use(&permit)?;
        assert!(matches!(
            scheduler
                .admissions
                .get(&first.admission_id)
                .map(|admission| &admission.state),
            Some(AdmissionState::Quarantined {
                resident: Some(resident),
            }) if resident.as_str() == "resident-first"
        ));
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
    fn native_resident_and_v1_leases_compete_without_double_charging() -> Result<(), SchedulerError>
    {
        let grant = request(&format!("[{}]", workload("legacy", 4)), 8)?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let _legacy = one_ticket(&mut scheduler, &grant)?;
        let _native = native_loaded(&mut scheduler, 4, "native-main")?;
        assert!(matches!(
            scheduler.admit_native_resident(NativeResidentRequest::new(
                "w7900",
                requested_device_bytes(1)?,
            )),
            Err(SchedulerError::NativeDeviceBytes { .. })
        ));
        assert_eq!(scheduler.admissions.len(), 2);
        Ok(())
    }

    #[test]
    fn native_capabilities_are_controller_local_and_load_quarantine_retains_charge()
    -> Result<(), SchedulerError> {
        let grant = request("[]", 4)?;
        let mut first = Scheduler::new(&grant, SchedulerLimits::default())?;
        let mut second = Scheduler::new(&grant, SchedulerLimits::default())?;
        let (ticket, load) = first.admit_native_resident(NativeResidentRequest::new(
            "w7900",
            requested_device_bytes(4)?,
        ))?;
        assert!(matches!(
            second.quarantine_native_load(&load),
            Err(SchedulerError::ForeignCapability { .. })
        ));
        first.quarantine_native_load(&load)?;
        assert!(matches!(
            first.begin_native_use(
                &ticket,
                NativeUseRequest::new(requested_device_bytes(1)?, None, None),
            ),
            Err(SchedulerError::AdmissionNotResident { .. })
        ));
        assert!(matches!(first.poll_command()?, PollOutcome::Idle));
        assert!(matches!(
            first.admit_native_resident(NativeResidentRequest::new(
                "w7900",
                requested_device_bytes(1)?,
            )),
            Err(SchedulerError::NativeDeviceBytes { .. })
        ));
        first.revoke(&first.generation())?;
        assert!(matches!(first.poll_command()?, PollOutcome::Idle));
        Ok(())
    }

    #[test]
    fn native_use_releases_mutable_peak_and_retains_host_result() -> Result<(), SchedulerError> {
        let grant = request("[]", 12)?;
        let mut scheduler = Scheduler::new_with_native_host_results(
            &grant,
            SchedulerLimits::try_new(64, 8, 1)?,
            NativeHostResultEnvelope::new(4),
        )?;
        let ticket = native_loaded(&mut scheduler, 4, "native-main")?;
        let first = scheduler.begin_native_use(
            &ticket,
            NativeUseRequest::new(
                requested_device_bytes(6)?,
                None,
                Some(requested_host_bytes(4)?),
            ),
        )?;
        assert!(matches!(
            scheduler.begin_native_use(
                &ticket,
                NativeUseRequest::new(
                    requested_device_bytes(1)?,
                    None,
                    Some(requested_host_bytes(1)?)
                ),
            ),
            Err(SchedulerError::NativeHostResultExhausted { .. })
        ));
        let result = scheduler.finish_native_use_after_teardown(&first)?.ok_or(
            SchedulerError::UnknownUsePermit {
                location: error_location(),
            },
        )?;
        assert!(matches!(
            scheduler.begin_native_use(
                &ticket,
                NativeUseRequest::new(requested_device_bytes(6)?, None, None),
            ),
            Err(SchedulerError::ActiveUseLimit { .. })
        ));
        scheduler.discard_native_result(&result)?;
        let second = scheduler.begin_native_use(
            &ticket,
            NativeUseRequest::new(requested_device_bytes(6)?, None, None),
        )?;
        assert!(
            scheduler
                .finish_native_use_after_teardown(&second)?
                .is_none(),
            "a use without retained output returns no result lease"
        );
        assert_eq!(scheduler.host_result_reserved, 0);
        Ok(())
    }

    #[test]
    fn retained_device_result_competes_until_explicit_discard() -> Result<(), SchedulerError> {
        let grant = request("[]", 12)?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let ticket = native_loaded(&mut scheduler, 4, "native-main")?;
        let use_permit = scheduler.begin_native_use(
            &ticket,
            NativeUseRequest::new(
                requested_device_bytes(4)?,
                Some(requested_device_bytes(4)?),
                None,
            ),
        )?;
        let result = scheduler
            .finish_native_use_after_teardown(&use_permit)?
            .ok_or(SchedulerError::UnknownUsePermit {
                location: error_location(),
            })?;
        assert!(matches!(
            scheduler.begin_native_use(
                &ticket,
                NativeUseRequest::new(requested_device_bytes(5)?, None, None),
            ),
            Err(SchedulerError::NativeDeviceBytes { .. })
        ));
        scheduler.discard_native_result(&result)?;
        let retry = scheduler.begin_native_use(
            &ticket,
            NativeUseRequest::new(requested_device_bytes(5)?, None, None),
        )?;
        assert!(
            scheduler
                .finish_native_use_after_teardown(&retry)?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn dropped_or_quarantined_native_use_keeps_all_charges() -> Result<(), SchedulerError> {
        let grant = request("[]", 12)?;
        let mut scheduler = Scheduler::new(&grant, SchedulerLimits::default())?;
        let ticket = native_loaded(&mut scheduler, 4, "native-main")?;
        let abandoned = scheduler.begin_native_use(
            &ticket,
            NativeUseRequest::new(requested_device_bytes(8)?, None, None),
        )?;
        drop(abandoned);
        assert!(matches!(scheduler.poll_command()?, PollOutcome::Idle));
        assert!(matches!(
            scheduler.begin_native_use(
                &ticket,
                NativeUseRequest::new(requested_device_bytes(1)?, None, None),
            ),
            Err(SchedulerError::NativeDeviceBytes { .. })
        ));

        let grant = request("[]", 12)?;
        let mut quarantined = Scheduler::new(&grant, SchedulerLimits::default())?;
        let ticket = native_loaded(&mut quarantined, 4, "native-quarantine")?;
        let permit = quarantined.begin_native_use(
            &ticket,
            NativeUseRequest::new(requested_device_bytes(4)?, None, None),
        )?;
        let finished = quarantined.begin_native_use(
            &ticket,
            NativeUseRequest::new(requested_device_bytes(1)?, None, None),
        )?;
        quarantined.revoke(&quarantined.generation())?;
        quarantined.quarantine_native_use(&permit)?;
        assert!(
            quarantined
                .finish_native_use_after_teardown(&finished)?
                .is_none(),
            "a known-finished sibling use releases only its mutable charge"
        );
        assert!(matches!(
            quarantined.begin_native_use(
                &ticket,
                NativeUseRequest::new(requested_device_bytes(1)?, None, None),
            ),
            Err(SchedulerError::AdmissionNotResident { .. })
        ));
        assert!(matches!(quarantined.poll_command()?, PollOutcome::Idle));
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
