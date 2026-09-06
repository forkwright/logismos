//! # placement
//!
//! Pure, CPU-only resource planning for one logismos artifact on one declared
//! accelerator. It consumes declared capacity and estimate inputs only; it
//! neither probes hardware nor reserves memory.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use isa::matches_configured_architecture;
use serde::{Deserialize, Serialize};
use snafu::Snafu;

/// Current JSON contract version.
pub const SCHEMA_VERSION: u32 = 1;

/// Resource-plan input after schema validation.
///
/// Construct this type by deserializing JSON. Its fields are intentionally
/// private so malformed external data cannot bypass the validation boundary.
/// Stable device, artifact, and profile IDs use the nonempty ASCII token
/// grammar `[A-Za-z0-9._:-]+`; it accepts PCI BDF punctuation while rejecting
/// whitespace, controls, path separators, and shell-like metacharacters.
#[derive(Debug, Clone, Serialize)]
pub struct PlanRequest {
    schema_version: u32,
    devices: Vec<Device>,
    artifacts: Vec<Artifact>,
    workloads: Vec<Workload>,
    commitments: Vec<DeviceCommitment>,
}

/// A declared accelerator available to the planner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Device {
    id: String,
    gfx_isa: String,
    total_bytes: u64,
    reserved_bytes: u64,
    availability: Availability,
}

/// Declared runtime availability; this is input fact, never a hardware probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Availability {
    /// The declared device may receive new placements.
    Available,
    /// The declared device must not receive new placements.
    Offline,
}

/// One immutable artifact identity in the declared catalogue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Artifact {
    artifact_id: String,
    digest: String,
}

/// One ordered workload profile that refers to an immutable artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Workload {
    profile_id: String,
    artifact_id: String,
    memory_estimate: MemoryEstimate,
    placement: PlacementRequest,
}

/// Explicit profile estimates used as a planning budget.
///
/// These values are estimates supplied by the profile author, not measured
/// VRAM use and not a physical reservation claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MemoryEstimate {
    #[serde(rename = "weights_bytes")]
    weights: u64,
    #[serde(rename = "kv_cache_bytes")]
    kv_cache: u64,
    #[serde(rename = "workspace_bytes")]
    workspace: u64,
    #[serde(rename = "headroom_bytes")]
    headroom: u64,
}

/// Requested placement policy for a workload profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlacementRequest {
    /// The workload requires this exact declared device.
    RequestedDevice {
        /// Stable ID of the required declared device.
        device_id: String,
    },
    /// The workload may use the first admitted device in this declared order.
    EligibleDevices {
        /// Stable device IDs, ranked by the profile author's preference.
        device_ids: Vec<String>,
    },
}

/// Existing estimated budget already committed on one declared device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCommitment {
    device_id: String,
    estimated_bytes: u64,
}

/// Deterministic result of a placement attempt.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PlanOutcome {
    /// Every workload was admitted transactionally.
    Plan {
        /// Contract version used for the response.
        schema_version: u32,
        /// One admitted placement per workload, in workload input order.
        admitted_placements: Vec<AdmittedPlacement>,
    },
    /// No allocation plan is applicable.
    Refusal {
        /// Contract version used for the response.
        schema_version: u32,
        /// Machine-readable reason that no plan was produced.
        refusal: PlacementRefusal,
    },
}

/// One workload's admitted per-device estimate.
#[derive(Debug, Clone, Serialize)]
pub struct AdmittedPlacement {
    /// Unique workload profile ID from the request.
    pub profile_id: String,
    /// Immutable artifact identity from the catalogue.
    pub artifact_id: String,
    /// Immutable artifact digest from the request.
    pub digest: String,
    /// Selected declared device.
    pub device_id: String,
    /// Profile estimate used to admit this one-device placement.
    pub memory_estimate: MemoryEstimate,
    /// Checked sum of the estimate breakdown.
    pub total_estimated_bytes: u64,
}

/// Mutable accounting authority for one immutable declared resource grant.
///
/// A ledger pins devices, capacities, availability, and static commitments at
/// construction. Later requests may change only their requested artifacts and
/// workloads; they cannot inflate or replace resource facts.
///
/// This is process-local accounting, not exclusive host ownership. The future
/// service owner must create exactly one ledger for each granted resource set
/// and reconcile it during shutdown or restart.
#[derive(Debug)]
pub struct ReservationLedger {
    brand: Arc<LedgerBrand>,
    snapshot: ResourceSnapshot,
    dynamic_reserved: BTreeMap<String, u64>,
    active_leases: BTreeMap<u64, LeaseRecord>,
    revision: u64,
    next_lease_id: u64,
}

/// A prepared batch that only its originating [`ReservationLedger`] can commit.
///
/// This is intentionally not serializable or deserializable: it is an
/// in-process capability, not a planning report.
#[derive(Debug)]
pub struct PreparedPlan {
    brand: Arc<LedgerBrand>,
    revision: u64,
    schema_version: u32,
    placements: Vec<AdmittedPlacement>,
}

/// One committed workload reservation.
///
/// This capability is consumed by [`ReservationLedger::release`], preventing
/// a caller from releasing the same reservation twice.
#[derive(Debug)]
pub struct ReservationLease {
    brand: Arc<LedgerBrand>,
    lease_id: u64,
    placement: AdmittedPlacement,
}

/// A failed lease release that returns the unconsumed capability to its owner.
///
/// Retaining the lease on failure lets a higher-level transaction restore its
/// own state without inventing a replacement reservation token.
#[derive(Debug)]
pub struct LeaseReleaseFailure {
    reason: PlacementRefusal,
    lease: ReservationLease,
}

#[derive(Debug)]
struct LedgerBrand;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResourceSnapshot {
    devices: Vec<Device>,
    commitments: Vec<DeviceCommitment>,
}

#[derive(Debug, Clone)]
struct LeaseRecord {
    device_id: String,
    reserved_bytes: u64,
}

/// Typed reason a request cannot yield a plan.
#[derive(Debug, Clone, Serialize, Snafu)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PlacementRefusal {
    /// The JSON did not deserialize into this contract's validated schema.
    #[snafu(display("input does not match the placement contract"))]
    InvalidRequest,
    /// The request named an unsupported contract version.
    #[snafu(display("unsupported schema version {found}"))]
    UnsupportedSchemaVersion {
        /// Version supplied by the request.
        found: u32,
    },
    /// No declared device was supplied.
    #[snafu(display("request has no declared devices"))]
    MissingDevices,
    /// A required string field was empty or whitespace-only.
    #[snafu(display("{field} must not be empty"))]
    EmptyField {
        /// Name of the invalid field.
        field: &'static str,
    },
    /// An artifact digest is not the canonical immutable SHA-256 form.
    #[snafu(display(
        "artifact digest must be sha256 followed by 64 lowercase hexadecimal characters"
    ))]
    InvalidDigest,
    /// A stable ID contains characters outside the safe token grammar.
    #[snafu(display("{field} must use ASCII letters, digits, '.', '_', '-', or ':' only"))]
    InvalidStableId {
        /// Name of the invalid field.
        field: &'static str,
    },
    /// A declared device ID appeared more than once.
    #[snafu(display("duplicate device {device_id}"))]
    DuplicateDevice {
        /// Repeated device ID.
        device_id: String,
    },
    /// An artifact catalogue ID appeared more than once.
    #[snafu(display("duplicate artifact {artifact_id}"))]
    DuplicateArtifact {
        /// Repeated artifact ID.
        artifact_id: String,
    },
    /// A workload profile ID appeared more than once.
    #[snafu(display("duplicate profile {profile_id}"))]
    DuplicateProfile {
        /// Repeated workload profile ID.
        profile_id: String,
    },
    /// A device commitment appeared more than once for one device.
    #[snafu(display("duplicate commitment for device {device_id}"))]
    DuplicateCommitment {
        /// Device with more than one commitment record.
        device_id: String,
    },
    /// A workload's eligible-device list repeated an ID.
    #[snafu(display("profile {profile_id} repeats eligible device {device_id}"))]
    DuplicateEligibleDevice {
        /// Profile whose placement list is invalid.
        profile_id: String,
        /// Repeated eligible device ID.
        device_id: String,
    },
    /// A workload references an absent immutable artifact catalogue entry.
    #[snafu(display("profile {profile_id} references missing artifact {artifact_id}"))]
    MissingArtifact {
        /// Workload profile that made the reference.
        profile_id: String,
        /// Absent artifact catalogue ID.
        artifact_id: String,
    },
    /// An explicitly requested device is absent from the declaration.
    #[snafu(display("profile {profile_id} requested missing device {device_id}"))]
    MissingDevice {
        /// Workload profile requiring the device.
        profile_id: String,
        /// Missing declared device ID.
        device_id: String,
    },
    /// A commitment names a device absent from the declaration.
    #[snafu(display("commitment names missing device {device_id}"))]
    CommitmentForMissingDevice {
        /// Missing declared device ID.
        device_id: String,
    },
    /// A device's ISA is outside this first-slice contract.
    #[snafu(display("device {device_id} has unsupported ISA {gfx_isa}"))]
    UnsupportedDevice {
        /// Unsupported declared device.
        device_id: String,
        /// Declared ISA.
        gfx_isa: String,
    },
    /// An explicitly requested device is not available for new placement.
    #[snafu(display("device {device_id} is offline"))]
    OfflineDevice {
        /// Offline declared device ID.
        device_id: String,
    },
    /// No eligible device was declared, online, supported, and large enough.
    #[snafu(display("profile {profile_id} has no eligible device"))]
    NoEligibleDevice {
        /// Workload profile that could not be admitted.
        profile_id: String,
    },
    /// A budget does not fit on its one selected device.
    #[snafu(display(
        "device {device_id} is exhausted: needs {required_bytes}, has {available_bytes}"
    ))]
    CapacityExhausted {
        /// Device whose individual capacity was insufficient.
        device_id: String,
        /// Required estimated bytes.
        required_bytes: u64,
        /// Remaining estimated bytes on that one device.
        available_bytes: u64,
    },
    /// Checked byte arithmetic overflowed.
    #[snafu(display("byte arithmetic overflow while computing {scope}"))]
    ArithmeticOverflow {
        /// Calculation that overflowed.
        scope: &'static str,
    },
    /// A later request attempted to replace the ledger's resource-grant facts.
    #[snafu(display("request resource snapshot differs from the ledger grant"))]
    ResourceSnapshotMismatch,
    /// A prepared batch belongs to a different ledger instance.
    #[snafu(display("prepared plan belongs to another reservation ledger"))]
    ForeignPreparedPlan,
    /// A prepared batch no longer reflects this ledger's current reservations.
    #[snafu(display("prepared plan is stale for the current reservation ledger"))]
    StalePreparedPlan,
    /// A lease belongs to a different ledger instance.
    #[snafu(display("reservation lease belongs to another reservation ledger"))]
    ForeignReservationLease,
    /// A lease is absent or does not match this ledger's active reservation.
    #[snafu(display("reservation lease is not active in this ledger"))]
    UnknownReservationLease,
}

/// Deserialize JSON and return a typed plan or refusal.
///
/// A malformed request has no usable schema version, so its refusal reports
/// this crate's current contract version.
#[must_use]
pub fn plan_json(input: &str) -> PlanOutcome {
    match serde_json::from_str::<RawPlanRequest>(input) {
        Ok(raw) => match PlanRequest::try_from(raw) {
            Ok(request) => plan(&request),
            Err(reason) => refusal(reason),
        },
        Err(_) => refusal(PlacementRefusal::InvalidRequest),
    }
}

/// Produce a plan from a checked request without any hardware interaction.
///
/// Workloads are considered in their input order. Eligible-device candidates
/// are considered in their declared order, so this deterministic first-fit
/// policy may refuse a batch that a global packing solver could place.
#[must_use]
pub fn plan(request: &PlanRequest) -> PlanOutcome {
    match ReservationLedger::new(request).and_then(|ledger| ledger.prepare(request)) {
        Ok(prepared) => prepared.into_outcome(),
        Err(reason) => refusal(reason),
    }
}

impl PlanRequest {
    /// Return the bounded request workload count without planning it.
    #[must_use]
    pub fn workload_count(&self) -> usize {
        self.workloads.len()
    }

    /// Parse and validate a v1 planning request without producing a report.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal when the input is malformed or violates the
    /// placement contract.
    pub fn from_json(input: &str) -> Result<Self, PlacementRefusal> {
        let raw = serde_json::from_str::<RawPlanRequest>(input)
            .map_err(|_| PlacementRefusal::InvalidRequest)?;
        Self::try_from(raw)
    }
}

impl ReservationLedger {
    /// Create process-local accounting for one validated declared resource grant.
    ///
    /// # Errors
    ///
    /// Returns a refusal when static reserved or committed bytes already
    /// exceed a declared device capacity.
    pub fn new(request: &PlanRequest) -> Result<Self, PlacementRefusal> {
        let snapshot = ResourceSnapshot::from_request(request);
        let _remaining = remaining_after_reservations(&snapshot, &BTreeMap::new())?;
        Ok(Self {
            brand: Arc::new(LedgerBrand),
            snapshot,
            dynamic_reserved: BTreeMap::new(),
            active_leases: BTreeMap::new(),
            revision: 0,
            next_lease_id: 1,
        })
    }

    /// Prepare a transactional batch against this ledger's current revision.
    ///
    /// # Errors
    ///
    /// Returns a refusal if the request changes resource-grant facts or if its
    /// requested work cannot fit beside current reservations.
    pub fn prepare(&self, request: &PlanRequest) -> Result<PreparedPlan, PlacementRefusal> {
        if self.snapshot != ResourceSnapshot::from_request(request) {
            return Err(PlacementRefusal::ResourceSnapshotMismatch);
        }
        let remaining = remaining_after_reservations(&self.snapshot, &self.dynamic_reserved)?;
        let placements = prepare_placements(request, remaining)?;
        Ok(PreparedPlan {
            brand: Arc::clone(&self.brand),
            revision: self.revision,
            schema_version: request.schema_version,
            placements,
        })
    }

    /// Atomically reserve every placement in a prepared batch.
    ///
    /// # Errors
    ///
    /// Returns a refusal without mutating this ledger if the prepared batch
    /// belongs to another ledger, is stale, overflows an identifier, or no
    /// longer fits current reservations.
    pub fn commit(
        &mut self,
        prepared: PreparedPlan,
    ) -> Result<Vec<ReservationLease>, PlacementRefusal> {
        if !Arc::ptr_eq(&self.brand, &prepared.brand) {
            return Err(PlacementRefusal::ForeignPreparedPlan);
        }
        if self.revision != prepared.revision {
            return Err(PlacementRefusal::StalePreparedPlan);
        }

        let lease_count = u64::try_from(prepared.placements.len()).map_err(|_| {
            PlacementRefusal::ArithmeticOverflow {
                scope: "prepared placement count",
            }
        })?;
        let _next_lease_id = self.next_lease_id.checked_add(lease_count).ok_or(
            PlacementRefusal::ArithmeticOverflow {
                scope: "reservation lease identifier",
            },
        )?;
        let next_revision =
            self.revision
                .checked_add(1)
                .ok_or(PlacementRefusal::ArithmeticOverflow {
                    scope: "reservation ledger revision",
                })?;

        let mut candidate_reserved = self.dynamic_reserved.clone();
        for placement in &prepared.placements {
            let reserved = candidate_reserved
                .entry(placement.device_id.clone())
                .or_default();
            *reserved = reserved
                .checked_add(placement.total_estimated_bytes)
                .ok_or(PlacementRefusal::ArithmeticOverflow {
                    scope: "dynamic device reservation",
                })?;
        }
        let _remaining = remaining_after_reservations(&self.snapshot, &candidate_reserved)?;

        let mut leases = Vec::with_capacity(prepared.placements.len());
        let mut next_lease_id = self.next_lease_id;
        for placement in prepared.placements {
            let lease_id = next_lease_id;
            next_lease_id =
                next_lease_id
                    .checked_add(1)
                    .ok_or(PlacementRefusal::ArithmeticOverflow {
                        scope: "reservation lease identifier",
                    })?;
            self.active_leases.insert(
                lease_id,
                LeaseRecord {
                    device_id: placement.device_id.clone(),
                    reserved_bytes: placement.total_estimated_bytes,
                },
            );
            leases.push(ReservationLease {
                brand: Arc::clone(&self.brand),
                lease_id,
                placement,
            });
        }
        self.dynamic_reserved = candidate_reserved;
        self.next_lease_id = next_lease_id;
        self.revision = next_revision;
        Ok(leases)
    }

    /// Release one committed reservation.
    ///
    /// # Errors
    ///
    /// Returns the typed refusal and original lease without mutation when the
    /// lease belongs to another ledger or was not active in this ledger.
    pub fn release(&mut self, lease: ReservationLease) -> Result<(), Box<LeaseReleaseFailure>> {
        if !Arc::ptr_eq(&self.brand, &lease.brand) {
            return Err(Box::new(LeaseReleaseFailure::new(
                PlacementRefusal::ForeignReservationLease,
                lease,
            )));
        }
        let Some(record) = self.active_leases.get(&lease.lease_id) else {
            return Err(Box::new(LeaseReleaseFailure::new(
                PlacementRefusal::UnknownReservationLease,
                lease,
            )));
        };
        if record.device_id != lease.placement.device_id
            || record.reserved_bytes != lease.placement.total_estimated_bytes
        {
            return Err(Box::new(LeaseReleaseFailure::new(
                PlacementRefusal::UnknownReservationLease,
                lease,
            )));
        }

        let Some(current_reserved) = self.dynamic_reserved.get(record.device_id.as_str()) else {
            return Err(Box::new(LeaseReleaseFailure::new(
                PlacementRefusal::UnknownReservationLease,
                lease,
            )));
        };
        let Some(next_reserved) = current_reserved.checked_sub(record.reserved_bytes) else {
            return Err(Box::new(LeaseReleaseFailure::new(
                PlacementRefusal::ArithmeticOverflow {
                    scope: "dynamic device release",
                },
                lease,
            )));
        };
        let Some(next_revision) = self.revision.checked_add(1) else {
            return Err(Box::new(LeaseReleaseFailure::new(
                PlacementRefusal::ArithmeticOverflow {
                    scope: "reservation ledger revision",
                },
                lease,
            )));
        };

        let device_id = record.device_id.clone();
        self.active_leases.remove(&lease.lease_id);
        if next_reserved == 0 {
            self.dynamic_reserved.remove(&device_id);
        } else {
            self.dynamic_reserved.insert(device_id, next_reserved);
        }
        self.revision = next_revision;
        Ok(())
    }
}

impl LeaseReleaseFailure {
    fn new(reason: PlacementRefusal, lease: ReservationLease) -> Self {
        Self { reason, lease }
    }

    /// Return the typed reason and the still-live lease capability.
    #[must_use]
    pub fn into_parts(self) -> (PlacementRefusal, ReservationLease) {
        (self.reason, self.lease)
    }
}

impl PreparedPlan {
    /// Return how many individual workload reservations this batch contains.
    #[must_use]
    pub fn placement_count(&self) -> usize {
        self.placements.len()
    }

    /// Convert this prepared batch into the stable v1 planning report.
    #[must_use]
    pub fn into_outcome(self) -> PlanOutcome {
        PlanOutcome::Plan {
            schema_version: self.schema_version,
            admitted_placements: self.placements,
        }
    }
}

impl ReservationLease {
    /// Return the validated profile ID this lease reserves.
    #[must_use]
    pub fn profile_id(&self) -> &str {
        &self.placement.profile_id
    }

    /// Return the immutable artifact ID this lease reserves.
    #[must_use]
    pub fn artifact_id(&self) -> &str {
        &self.placement.artifact_id
    }

    /// Return the immutable artifact digest this lease reserves.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.placement.digest
    }

    /// Return the declared device ID this lease reserves.
    #[must_use]
    pub fn device_id(&self) -> &str {
        &self.placement.device_id
    }

    /// Return the checked estimate that was reserved for this lease.
    #[must_use]
    pub fn total_estimated_bytes(&self) -> u64 {
        self.placement.total_estimated_bytes
    }
}

impl ResourceSnapshot {
    fn from_request(request: &PlanRequest) -> Self {
        Self {
            devices: request.devices.clone(),
            commitments: request.commitments.clone(),
        }
    }
}

fn prepare_placements(
    request: &PlanRequest,
    mut remaining: Vec<u64>,
) -> Result<Vec<AdmittedPlacement>, PlacementRefusal> {
    let mut device_indices = BTreeMap::new();
    for (index, device) in request.devices.iter().enumerate() {
        let _previous = device_indices.insert(device.id.as_str(), index);
    }

    let mut admitted_placements = Vec::with_capacity(request.workloads.len());
    let artifacts = request
        .artifacts
        .iter()
        .map(|artifact| (artifact.artifact_id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    for workload in &request.workloads {
        let Some(artifact) = artifacts.get(workload.artifact_id.as_str()) else {
            return Err(PlacementRefusal::MissingArtifact {
                profile_id: workload.profile_id.clone(),
                artifact_id: workload.artifact_id.clone(),
            });
        };
        let required_bytes = workload.memory_estimate.total()?;
        let device_index = select_device(
            workload,
            &request.devices,
            &device_indices,
            &remaining,
            required_bytes,
        )?;
        remaining[device_index] -= required_bytes;
        admitted_placements.push(AdmittedPlacement {
            profile_id: workload.profile_id.clone(),
            artifact_id: artifact.artifact_id.clone(),
            digest: artifact.digest.clone(),
            device_id: request.devices[device_index].id.clone(),
            memory_estimate: workload.memory_estimate,
            total_estimated_bytes: required_bytes,
        });
    }

    Ok(admitted_placements)
}

fn remaining_after_reservations(
    snapshot: &ResourceSnapshot,
    dynamic_reserved: &BTreeMap<String, u64>,
) -> Result<Vec<u64>, PlacementRefusal> {
    let mut device_indices = BTreeMap::new();
    for (index, device) in snapshot.devices.iter().enumerate() {
        let _previous = device_indices.insert(device.id.as_str(), index);
    }

    let mut remaining = Vec::with_capacity(snapshot.devices.len());
    for device in &snapshot.devices {
        let Some(after_reserved) = device.total_bytes.checked_sub(device.reserved_bytes) else {
            return Err(PlacementRefusal::CapacityExhausted {
                device_id: device.id.clone(),
                required_bytes: device.reserved_bytes,
                available_bytes: device.total_bytes,
            });
        };
        remaining.push(after_reserved);
    }
    for commitment in &snapshot.commitments {
        let Some(&index) = device_indices.get(commitment.device_id.as_str()) else {
            return Err(PlacementRefusal::CommitmentForMissingDevice {
                device_id: commitment.device_id.clone(),
            });
        };
        let Some(after_commitment) = remaining[index].checked_sub(commitment.estimated_bytes)
        else {
            return Err(PlacementRefusal::CapacityExhausted {
                device_id: commitment.device_id.clone(),
                required_bytes: commitment.estimated_bytes,
                available_bytes: remaining[index],
            });
        };
        remaining[index] = after_commitment;
    }
    for (device_id, reserved_bytes) in dynamic_reserved {
        let Some(&index) = device_indices.get(device_id.as_str()) else {
            return Err(PlacementRefusal::CommitmentForMissingDevice {
                device_id: device_id.clone(),
            });
        };
        let Some(after_dynamic) = remaining[index].checked_sub(*reserved_bytes) else {
            return Err(PlacementRefusal::CapacityExhausted {
                device_id: device_id.clone(),
                required_bytes: *reserved_bytes,
                available_bytes: remaining[index],
            });
        };
        remaining[index] = after_dynamic;
    }
    Ok(remaining)
}

fn select_device(
    workload: &Workload,
    devices: &[Device],
    device_indices: &BTreeMap<&str, usize>,
    remaining: &[u64],
    required_bytes: u64,
) -> Result<usize, PlacementRefusal> {
    match &workload.placement {
        PlacementRequest::RequestedDevice { device_id } => {
            let Some(&index) = device_indices.get(device_id.as_str()) else {
                return Err(PlacementRefusal::MissingDevice {
                    profile_id: workload.profile_id.clone(),
                    device_id: device_id.clone(),
                });
            };
            let device = &devices[index];
            if !matches_configured_architecture(&device.gfx_isa) {
                return Err(PlacementRefusal::UnsupportedDevice {
                    device_id: device.id.clone(),
                    gfx_isa: device.gfx_isa.clone(),
                });
            }
            if device.availability == Availability::Offline {
                return Err(PlacementRefusal::OfflineDevice {
                    device_id: device.id.clone(),
                });
            }
            if remaining[index] < required_bytes {
                return Err(PlacementRefusal::CapacityExhausted {
                    device_id: device.id.clone(),
                    required_bytes,
                    available_bytes: remaining[index],
                });
            }
            Ok(index)
        }
        PlacementRequest::EligibleDevices { device_ids } => {
            for device_id in device_ids {
                let Some(&index) = device_indices.get(device_id.as_str()) else {
                    continue;
                };
                let device = &devices[index];
                if matches_configured_architecture(&device.gfx_isa)
                    && device.availability == Availability::Available
                    && remaining[index] >= required_bytes
                {
                    return Ok(index);
                }
            }
            Err(PlacementRefusal::NoEligibleDevice {
                profile_id: workload.profile_id.clone(),
            })
        }
    }
}

fn refusal(reason: PlacementRefusal) -> PlanOutcome {
    PlanOutcome::Refusal {
        schema_version: SCHEMA_VERSION,
        refusal: reason,
    }
}

impl MemoryEstimate {
    fn total(self) -> Result<u64, PlacementRefusal> {
        self.weights
            .checked_add(self.kv_cache)
            .and_then(|bytes| bytes.checked_add(self.workspace))
            .and_then(|bytes| bytes.checked_add(self.headroom))
            .ok_or(PlacementRefusal::ArithmeticOverflow {
                scope: "workload memory estimate",
            })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlanRequest {
    schema_version: u32,
    devices: Vec<RawDevice>,
    artifacts: Vec<RawArtifact>,
    workloads: Vec<RawWorkload>,
    #[serde(default)]
    commitments: Vec<RawDeviceCommitment>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDevice {
    id: String,
    gfx_isa: String,
    total_bytes: u64,
    reserved_bytes: u64,
    availability: Availability,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    artifact_id: String,
    digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMemoryEstimate {
    #[serde(rename = "weights_bytes")]
    weights: u64,
    #[serde(rename = "kv_cache_bytes")]
    kv_cache: u64,
    #[serde(rename = "workspace_bytes")]
    workspace: u64,
    #[serde(rename = "headroom_bytes")]
    headroom: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkload {
    profile_id: String,
    artifact_id: String,
    memory_estimate: RawMemoryEstimate,
    placement: RawPlacementRequest,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum RawPlacementRequest {
    RequestedDevice { device_id: String },
    EligibleDevices { device_ids: Vec<String> },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDeviceCommitment {
    device_id: String,
    estimated_bytes: u64,
}

impl<'de> Deserialize<'de> for PlanRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawPlanRequest::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

impl TryFrom<RawPlanRequest> for PlanRequest {
    type Error = PlacementRefusal;

    fn try_from(raw: RawPlanRequest) -> Result<Self, Self::Error> {
        if raw.schema_version != SCHEMA_VERSION {
            return Err(PlacementRefusal::UnsupportedSchemaVersion {
                found: raw.schema_version,
            });
        }
        if raw.devices.is_empty() {
            return Err(PlacementRefusal::MissingDevices);
        }

        let devices = parse_devices(raw.devices)?;
        let artifacts = parse_artifacts(raw.artifacts)?;
        let workloads = parse_workloads(raw.workloads)?;
        let commitments = parse_commitments(raw.commitments)?;

        Ok(Self {
            schema_version: raw.schema_version,
            devices,
            artifacts,
            workloads,
            commitments,
        })
    }
}

fn parse_devices(raw_devices: Vec<RawDevice>) -> Result<Vec<Device>, PlacementRefusal> {
    let mut ids = BTreeSet::new();
    raw_devices
        .into_iter()
        .map(|raw_device| {
            require_stable_id("device.id", &raw_device.id)?;
            require_non_empty("device.gfx_isa", &raw_device.gfx_isa)?;
            if !ids.insert(raw_device.id.clone()) {
                return Err(PlacementRefusal::DuplicateDevice {
                    device_id: raw_device.id,
                });
            }
            Ok(Device {
                id: raw_device.id,
                gfx_isa: raw_device.gfx_isa,
                total_bytes: raw_device.total_bytes,
                reserved_bytes: raw_device.reserved_bytes,
                availability: raw_device.availability,
            })
        })
        .collect()
}

fn parse_artifacts(raw_artifacts: Vec<RawArtifact>) -> Result<Vec<Artifact>, PlacementRefusal> {
    let mut ids = BTreeSet::new();
    raw_artifacts
        .into_iter()
        .map(|raw_artifact| {
            require_stable_id("artifact.artifact_id", &raw_artifact.artifact_id)?;
            require_non_empty("artifact.digest", &raw_artifact.digest)?;
            validate_digest(&raw_artifact.digest)?;
            if !ids.insert(raw_artifact.artifact_id.clone()) {
                return Err(PlacementRefusal::DuplicateArtifact {
                    artifact_id: raw_artifact.artifact_id,
                });
            }
            Ok(Artifact {
                artifact_id: raw_artifact.artifact_id,
                digest: raw_artifact.digest,
            })
        })
        .collect()
}

fn parse_workloads(raw_workloads: Vec<RawWorkload>) -> Result<Vec<Workload>, PlacementRefusal> {
    let mut ids = BTreeSet::new();
    raw_workloads
        .into_iter()
        .map(|raw_workload| {
            require_stable_id("workload.profile_id", &raw_workload.profile_id)?;
            require_stable_id("workload.artifact_id", &raw_workload.artifact_id)?;
            if !ids.insert(raw_workload.profile_id.clone()) {
                return Err(PlacementRefusal::DuplicateProfile {
                    profile_id: raw_workload.profile_id,
                });
            }
            let placement = validate_placement(&raw_workload)?;
            let memory_estimate = MemoryEstimate {
                weights: raw_workload.memory_estimate.weights,
                kv_cache: raw_workload.memory_estimate.kv_cache,
                workspace: raw_workload.memory_estimate.workspace,
                headroom: raw_workload.memory_estimate.headroom,
            };
            let _total = memory_estimate.total()?;
            Ok(Workload {
                profile_id: raw_workload.profile_id,
                artifact_id: raw_workload.artifact_id,
                memory_estimate,
                placement,
            })
        })
        .collect()
}

fn parse_commitments(
    raw_commitments: Vec<RawDeviceCommitment>,
) -> Result<Vec<DeviceCommitment>, PlacementRefusal> {
    let mut ids = BTreeSet::new();
    raw_commitments
        .into_iter()
        .map(|raw_commitment| {
            require_stable_id("commitment.device_id", &raw_commitment.device_id)?;
            if !ids.insert(raw_commitment.device_id.clone()) {
                return Err(PlacementRefusal::DuplicateCommitment {
                    device_id: raw_commitment.device_id,
                });
            }
            Ok(DeviceCommitment {
                device_id: raw_commitment.device_id,
                estimated_bytes: raw_commitment.estimated_bytes,
            })
        })
        .collect()
}

fn validate_placement(raw: &RawWorkload) -> Result<PlacementRequest, PlacementRefusal> {
    match &raw.placement {
        RawPlacementRequest::RequestedDevice { device_id } => {
            require_stable_id("workload.placement.device_id", device_id)?;
            Ok(PlacementRequest::RequestedDevice {
                device_id: device_id.clone(),
            })
        }
        RawPlacementRequest::EligibleDevices { device_ids } => {
            if device_ids.is_empty() {
                return Err(PlacementRefusal::NoEligibleDevice {
                    profile_id: raw.profile_id.clone(),
                });
            }
            let mut eligible_ids = BTreeSet::new();
            for device_id in device_ids {
                require_stable_id("workload.placement.device_ids", device_id)?;
                if !eligible_ids.insert(device_id.clone()) {
                    return Err(PlacementRefusal::DuplicateEligibleDevice {
                        profile_id: raw.profile_id.clone(),
                        device_id: device_id.clone(),
                    });
                }
            }
            Ok(PlacementRequest::EligibleDevices {
                device_ids: device_ids.clone(),
            })
        }
    }
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), PlacementRefusal> {
    if value.trim().is_empty() {
        return Err(PlacementRefusal::EmptyField { field });
    }
    Ok(())
}

fn require_stable_id(field: &'static str, value: &str) -> Result<(), PlacementRefusal> {
    require_non_empty(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':'))
    {
        return Err(PlacementRefusal::InvalidStableId { field });
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), PlacementRefusal> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(PlacementRefusal::InvalidDigest);
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PlacementRefusal::InvalidDigest);
    }
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    const DIGEST_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn plan_input(devices: &str, artifacts: &str, workloads: &str) -> String {
        format!(
            r#"{{"schema_version":1,"devices":{devices},"artifacts":{artifacts},"workloads":{workloads},"commitments":[]}}"#
        )
    }

    #[test]
    fn gpu_boundary_pure_planner_uses_shared_isa_contract() {
        let outcome_for = |isa: &str| {
            let devices = format!(
                r#"[{{"id":"w7900","gfx_isa":"{isa}","total_bytes":48,"reserved_bytes":0,"availability":"available"}}]"#
            );
            let artifacts = format!(r#"[{{"artifact_id":"model","digest":"{DIGEST_A}"}}]"#);
            let workloads = r#"[{"profile_id":"main","artifact_id":"model","memory_estimate":{"weights_bytes":1,"kv_cache_bytes":0,"workspace_bytes":0,"headroom_bytes":0},"placement":{"kind":"requested_device","device_id":"w7900"}}]"#;
            plan_json(&plan_input(&devices, &artifacts, workloads))
        };

        let target = isa::configured_target_token();
        assert!(matches!(
            outcome_for(&format!("{target}:sramecc+:xnack-")),
            PlanOutcome::Plan { .. }
        ));
        assert!(matches!(
            outcome_for(&format!("{target}:xnack+:xnack-")),
            PlanOutcome::Refusal {
                refusal: PlacementRefusal::UnsupportedDevice { .. },
                ..
            }
        ));
    }

    #[test]
    fn places_profiles_in_workload_order_without_pooling() -> Result<(), PlacementRefusal> {
        let input = plan_input(
            r#"[
                {"id":"w7900","gfx_isa":"gfx1100","total_bytes":48,"reserved_bytes":8,"availability":"available"},
                {"id":"secondary-24gb","gfx_isa":"gfx1100","total_bytes":24,"reserved_bytes":0,"availability":"available"}
            ]"#,
            &format!(
                r#"[
                    {{"artifact_id":"decoder-artifact","digest":"{DIGEST_A}"}},
                    {{"artifact_id":"rerank-artifact","digest":"{DIGEST_B}"}}
                ]"#
            ),
            r#"[
                {"profile_id":"decoder-main","artifact_id":"decoder-artifact","memory_estimate":{"weights_bytes":20,"kv_cache_bytes":10,"workspace_bytes":4,"headroom_bytes":2},"placement":{"kind":"requested_device","device_id":"w7900"}},
                {"profile_id":"rerank-main","artifact_id":"rerank-artifact","memory_estimate":{"weights_bytes":8,"kv_cache_bytes":4,"workspace_bytes":1,"headroom_bytes":1},"placement":{"kind":"eligible_devices","device_ids":["optional-xtx","secondary-24gb"]}}
            ]"#,
        );

        let outcome = plan_json(&input);
        let admitted_placements = match outcome {
            PlanOutcome::Plan {
                admitted_placements,
                ..
            } => admitted_placements,
            PlanOutcome::Refusal { refusal, .. } => return Err(refusal),
        };
        assert_eq!(
            admitted_placements.len(),
            2,
            "each workload has one placement"
        );
        assert_eq!(
            admitted_placements[0].profile_id, "decoder-main",
            "input workload order is preserved"
        );
        assert_eq!(
            admitted_placements[0].device_id, "w7900",
            "explicit target is preserved"
        );
        assert_eq!(
            admitted_placements[1].device_id, "secondary-24gb",
            "absent optional XTX is skipped"
        );
        Ok(())
    }

    #[test]
    fn required_missing_device_and_missing_artifact_are_typed_refusals() {
        let missing_device = plan_input(
            r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":48,"reserved_bytes":0,"availability":"available"}]"#,
            &format!(r#"[{{"artifact_id":"head-artifact","digest":"{DIGEST_A}"}}]"#),
            r#"[{"profile_id":"head-main","artifact_id":"head-artifact","memory_estimate":{"weights_bytes":1,"kv_cache_bytes":1,"workspace_bytes":1,"headroom_bytes":1},"placement":{"kind":"requested_device","device_id":"optional-xtx"}}]"#,
        );
        assert!(
            matches!(
                plan_json(&missing_device),
                PlanOutcome::Refusal {
                    refusal: PlacementRefusal::MissingDevice { .. },
                    ..
                }
            ),
            "an explicit absent device is fatal"
        );

        let missing_artifact = plan_input(
            r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":48,"reserved_bytes":0,"availability":"available"}]"#,
            "[]",
            r#"[{"profile_id":"head-main","artifact_id":"missing-artifact","memory_estimate":{"weights_bytes":1,"kv_cache_bytes":1,"workspace_bytes":1,"headroom_bytes":1},"placement":{"kind":"eligible_devices","device_ids":["w7900"]}}]"#,
        );
        assert!(
            matches!(
                plan_json(&missing_artifact),
                PlanOutcome::Refusal {
                    refusal: PlacementRefusal::MissingArtifact { .. },
                    ..
                }
            ),
            "workloads must reference the immutable catalogue"
        );
    }

    #[test]
    fn one_workload_cannot_combine_48_and_24_capacity() {
        let input = plan_input(
            r#"[
                {"id":"w7900","gfx_isa":"gfx1100","total_bytes":48,"reserved_bytes":0,"availability":"available"},
                {"id":"secondary-24gb","gfx_isa":"gfx1100","total_bytes":24,"reserved_bytes":0,"availability":"available"}
            ]"#,
            &format!(r#"[{{"artifact_id":"large-artifact","digest":"{DIGEST_A}"}}]"#),
            r#"[{"profile_id":"large-main","artifact_id":"large-artifact","memory_estimate":{"weights_bytes":50,"kv_cache_bytes":0,"workspace_bytes":0,"headroom_bytes":0},"placement":{"kind":"eligible_devices","device_ids":["w7900","secondary-24gb"]}}]"#,
        );
        assert!(
            matches!(
                plan_json(&input),
                PlanOutcome::Refusal {
                    refusal: PlacementRefusal::NoEligibleDevice { .. },
                    ..
                }
            ),
            "the planner does not pool device budgets"
        );
    }

    #[test]
    fn failures_have_no_partial_plan_and_output_is_deterministic() -> Result<(), PlacementRefusal> {
        let input = plan_input(
            r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":10,"reserved_bytes":0,"availability":"available"}]"#,
            &format!(r#"[{{"artifact_id":"shared-artifact","digest":"{DIGEST_A}"}}]"#),
            r#"[
                {"profile_id":"first","artifact_id":"shared-artifact","memory_estimate":{"weights_bytes":6,"kv_cache_bytes":0,"workspace_bytes":0,"headroom_bytes":0},"placement":{"kind":"requested_device","device_id":"w7900"}},
                {"profile_id":"second","artifact_id":"shared-artifact","memory_estimate":{"weights_bytes":6,"kv_cache_bytes":0,"workspace_bytes":0,"headroom_bytes":0},"placement":{"kind":"requested_device","device_id":"w7900"}}
            ]"#,
        );
        assert!(
            matches!(
                plan_json(&input),
                PlanOutcome::Refusal {
                    refusal: PlacementRefusal::CapacityExhausted { .. },
                    ..
                }
            ),
            "failed transaction emits a refusal instead of a partial plan"
        );

        let stable_input = plan_input(
            r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":48,"reserved_bytes":0,"availability":"available"}]"#,
            &format!(r#"[{{"artifact_id":"head-artifact","digest":"{DIGEST_B}"}}]"#),
            r#"[{"profile_id":"head-main","artifact_id":"head-artifact","memory_estimate":{"weights_bytes":1,"kv_cache_bytes":1,"workspace_bytes":1,"headroom_bytes":1},"placement":{"kind":"eligible_devices","device_ids":["w7900"]}}]"#,
        );
        let first = serde_json::to_string(&plan_json(&stable_input))
            .map_err(|_| PlacementRefusal::InvalidRequest)?;
        let second = serde_json::to_string(&plan_json(&stable_input))
            .map_err(|_| PlacementRefusal::InvalidRequest)?;
        assert_eq!(first, second, "identical input emits identical JSON");
        Ok(())
    }

    #[test]
    fn rejects_unknown_fields_invalid_ids_and_overflow() {
        let unknown = r#"{"schema_version":1,"devices":[],"artifacts":[],"workloads":[],"commitments":[],"extra":true}"#;
        assert!(
            matches!(
                plan_json(unknown),
                PlanOutcome::Refusal {
                    refusal: PlacementRefusal::InvalidRequest,
                    ..
                }
            ),
            "schemas reject unknown fields"
        );

        let invalid_id = plan_input(
            r#"[{"id":"w7900/unsafe","gfx_isa":"gfx1100","total_bytes":48,"reserved_bytes":0,"availability":"available"}]"#,
            "[]",
            "[]",
        );
        assert!(
            matches!(
                plan_json(&invalid_id),
                PlanOutcome::Refusal {
                    refusal: PlacementRefusal::InvalidStableId { .. },
                    ..
                }
            ),
            "stable IDs reject slashes and controls with typed refusals"
        );

        let overflow = plan_input(
            r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":18446744073709551615,"reserved_bytes":0,"availability":"available"}]"#,
            &format!(r#"[{{"artifact_id":"overflow-artifact","digest":"{DIGEST_A}"}}]"#),
            r#"[{"profile_id":"overflow-main","artifact_id":"overflow-artifact","memory_estimate":{"weights_bytes":18446744073709551615,"kv_cache_bytes":1,"workspace_bytes":0,"headroom_bytes":0},"placement":{"kind":"requested_device","device_id":"w7900"}}]"#,
        );
        assert!(
            matches!(
                plan_json(&overflow),
                PlanOutcome::Refusal {
                    refusal: PlacementRefusal::ArithmeticOverflow { .. },
                    ..
                }
            ),
            "overflowing estimates are typed refusals"
        );
    }

    #[test]
    fn release_revision_overflow_returns_lease_without_mutating_accounting()
    -> Result<(), PlacementRefusal> {
        let input = plan_input(
            r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":10,"reserved_bytes":0,"availability":"available"}]"#,
            &format!(r#"[{{"artifact_id":"model","digest":"{DIGEST_A}"}}]"#),
            r#"[{"profile_id":"main","artifact_id":"model","memory_estimate":{"weights_bytes":4,"kv_cache_bytes":0,"workspace_bytes":0,"headroom_bytes":0},"placement":{"kind":"requested_device","device_id":"w7900"}}]"#,
        );
        let request = PlanRequest::from_json(&input)?;
        let mut ledger = ReservationLedger::new(&request)?;
        let prepared = ledger.prepare(&request)?;
        let mut leases = ledger.commit(prepared)?;
        let lease = leases
            .pop()
            .ok_or(PlacementRefusal::UnknownReservationLease)?;
        ledger.revision = u64::MAX;
        let failure = match ledger.release(lease) {
            Ok(()) => return Err(PlacementRefusal::UnknownReservationLease),
            Err(failure) => failure,
        };
        let (reason, lease) = failure.into_parts();
        assert!(
            matches!(reason, PlacementRefusal::ArithmeticOverflow { .. }),
            "counter exhaustion is a typed refusal"
        );
        assert_eq!(
            ledger.active_leases.len(),
            1,
            "failed release keeps the active lease"
        );
        assert_eq!(
            ledger.dynamic_reserved.get("w7900"),
            Some(&4),
            "failed release keeps reserved bytes unchanged"
        );
        drop(lease);
        Ok(())
    }

    #[test]
    fn ledger_rejects_later_resource_capacity_or_commitment_replacement()
    -> Result<(), PlacementRefusal> {
        let devices = r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":10,"reserved_bytes":0,"availability":"available"}]"#;
        let artifacts = format!(r#"[{{"artifact_id":"model","digest":"{DIGEST_A}"}}]"#);
        let workloads = r#"[{"profile_id":"main","artifact_id":"model","memory_estimate":{"weights_bytes":4,"kv_cache_bytes":0,"workspace_bytes":0,"headroom_bytes":0},"placement":{"kind":"requested_device","device_id":"w7900"}}]"#;
        let request = PlanRequest::from_json(&plan_input(devices, &artifacts, workloads))?;
        let ledger = ReservationLedger::new(&request)?;
        let altered_capacity = PlanRequest::from_json(&plan_input(
            r#"[{"id":"w7900","gfx_isa":"gfx1100","total_bytes":11,"reserved_bytes":0,"availability":"available"}]"#,
            &artifacts,
            workloads,
        ))?;
        assert!(
            matches!(
                ledger.prepare(&altered_capacity),
                Err(PlacementRefusal::ResourceSnapshotMismatch)
            ),
            "a later request cannot inflate the ledger's declared capacity"
        );
        let commitment_input = plan_input(devices, &artifacts, workloads).replace(
            "\"commitments\":[]",
            "\"commitments\":[{\"device_id\":\"w7900\",\"estimated_bytes\":1}]",
        );
        let altered_commitment = PlanRequest::from_json(&commitment_input)?;
        assert!(
            matches!(
                ledger.prepare(&altered_commitment),
                Err(PlacementRefusal::ResourceSnapshotMismatch)
            ),
            "a later request cannot replace static commitments"
        );
        Ok(())
    }
}
