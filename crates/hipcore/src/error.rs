//! Typed error surface for `hipcore`.
//!
//! Every FFI call that can fail produces an [`Error`] via the helper
//! [`check`] or [`check_with`]. Call-sites carry a static operation
//! name for diagnostic clarity.

use std::ffi::c_int;

use snafu::Snafu;

use crate::device::{DeviceUuid, PciBusId};
use crate::ffi;

/// Result alias used throughout `hipcore`.
pub type Result<T> = core::result::Result<T, Error>;

/// Every failure surface `hipcore` can emit.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// A HIP runtime call returned a non-success status.
    #[snafu(display("HIP runtime error: {kind:?} ({code}) in `{op}`"))]
    Runtime {
        /// Classified error kind.
        kind: ErrorKind,
        /// Raw HIP error code.
        code: u32,
        /// Static operation name.
        op: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Device ordinal out of range.
    #[snafu(display("device {ordinal} not found (host reports {count} devices)"))]
    NoSuchDevice {
        /// Requested ordinal.
        ordinal: c_int,
        /// Devices reported by `hipGetDeviceCount`.
        count: c_int,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A requested PCI-addressed device is not visible to HIP.
    #[snafu(display("device with PCI bus ID `{pci_bus_id}` not found"))]
    NoDeviceWithPciBusId {
        /// Stable PCI topology identifier requested by the caller.
        pci_bus_id: PciBusId,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A requested UUID-addressed device is not visible to HIP.
    #[snafu(display("device with UUID `{uuid}` not found"))]
    NoDeviceWithUuid {
        /// Stable UUID requested by the caller.
        uuid: DeviceUuid,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Two resources passed to one HIP operation belong to different devices.
    #[snafu(display("device mismatch in `{op}`: expected device {expected}, got device {actual}"))]
    DeviceMismatch {
        /// HIP operation or wrapper boundary that rejected the resources.
        op: &'static str,
        /// Ordinal of the resource that establishes the operation's device.
        expected: c_int,
        /// Ordinal of the incompatible resource.
        actual: c_int,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A PCI bus identifier was not in canonical `dddd:bb:dd.f` form.
    #[snafu(display("invalid PCI bus ID `{value}`; expected canonical `dddd:bb:dd.f` form"))]
    InvalidPciBusId {
        /// Rejected identifier.
        value: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Allocation failed because device memory is exhausted.
    #[snafu(display("out of device memory: requested {requested} bytes, {free} bytes free"))]
    OutOfMemory {
        /// Allocation size in bytes.
        requested: usize,
        /// Free VRAM at the time of the failed call.
        free: u64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Kernel launch failed; wraps the underlying HIP error.
    #[snafu(display("kernel launch failed: {kernel}: {source}"))]
    LaunchFailure {
        /// Kernel symbol name.
        kernel: &'static str,
        /// Source HIP error.
        source: Box<Error>,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Detected ISA does not match the configured target architecture.
    ///
    /// Non-fatal at construction time (a logismos consumer may choose
    /// to continue on another ISA), but surfaced so callers can decide.
    #[snafu(display("ISA `{isa}` does not match configured target architecture `{configured}`"))]
    UnsupportedIsa {
        /// ISA reported by `hipGetDeviceProperties`.
        isa: String,
        /// Full configured target token used for the architecture comparison.
        configured: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Something went wrong inside the safe wrapper rather than HIP.
    #[snafu(display("internal hipcore error: {message}"))]
    Internal {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Classified HIP error kind.
///
/// Maps the subset of `hipError_t` codes `hipcore` surfaces; the
/// `Unknown` carries unmapped codes when the caller has a raw numeric
/// status. FFI enum values outside the explicit map use a sentinel
/// because bindgen's rustified enum does not expose a checked numeric
/// conversion.
#[expect(
    missing_docs,
    reason = "variant names are self-documenting HIP runtime codes"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    InsufficientDriver,
    PriorLaunchFailure,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchFailure,
    NotFound,
    NotReady,
    IllegalAddress,
    /// Catch-all for unmapped raw codes.
    Unknown(u32),
}

impl ErrorKind {
    /// Map a raw HIP runtime error code to a classified kind.
    ///
    /// Public so other tier-1 crates (e.g. `kernels::Error::Launch`)
    /// can attach a symbolic name to a raw code they receive directly
    /// from a kernel launch, without re-deriving this mapping.
    #[must_use]
    pub fn from_raw(code: u32) -> Self {
        use ffi::hipError_t;
        match hipError_t_from_u32(code) {
            Some(hipError_t::hipErrorInvalidValue) => Self::InvalidValue,
            Some(hipError_t::hipErrorOutOfMemory) => Self::OutOfMemory,
            Some(hipError_t::hipErrorNotInitialized) => Self::NotInitialized,
            Some(hipError_t::hipErrorDeinitialized) => Self::Deinitialized,
            Some(hipError_t::hipErrorInsufficientDriver) => Self::InsufficientDriver,
            Some(hipError_t::hipErrorPriorLaunchFailure) => Self::PriorLaunchFailure,
            Some(hipError_t::hipErrorLaunchOutOfResources) => Self::LaunchOutOfResources,
            Some(hipError_t::hipErrorLaunchTimeOut) => Self::LaunchTimeout,
            Some(hipError_t::hipErrorLaunchFailure) => Self::LaunchFailure,
            Some(hipError_t::hipErrorNotFound) => Self::NotFound,
            Some(hipError_t::hipErrorNotReady) => Self::NotReady,
            Some(hipError_t::hipErrorIllegalAddress) => Self::IllegalAddress,
            _ => Self::Unknown(code),
        }
    }
}

#[expect(
    non_snake_case,
    reason = "matches HIP FFI type name `hipError_t` for call-site clarity"
)]
fn hipError_t_from_u32(code: u32) -> Option<ffi::hipError_t> {
    use ffi::hipError_t as E;
    // Explicit match — the enum is marked `rustified_enum` so we can't
    // mem::transmute safely. Covers the codes we surface; anything else
    // falls through to `Unknown(code)`.
    match code {
        0 => Some(E::hipSuccess),
        1 => Some(E::hipErrorInvalidValue),
        2 => Some(E::hipErrorOutOfMemory),
        3 => Some(E::hipErrorNotInitialized),
        4 => Some(E::hipErrorDeinitialized),
        35 => Some(E::hipErrorInsufficientDriver),
        53 => Some(E::hipErrorPriorLaunchFailure),
        701 => Some(E::hipErrorLaunchOutOfResources),
        702 => Some(E::hipErrorLaunchTimeOut),
        719 => Some(E::hipErrorLaunchFailure),
        500 => Some(E::hipErrorNotFound),
        600 => Some(E::hipErrorNotReady),
        700 => Some(E::hipErrorIllegalAddress),
        _ => None,
    }
}

#[expect(
    non_snake_case,
    reason = "matches HIP FFI type name `hipError_t` for call-site clarity"
)]
pub(crate) fn hipError_t_code(code: ffi::hipError_t) -> u32 {
    use ffi::hipError_t as E;
    match code {
        E::hipSuccess => 0,
        E::hipErrorInvalidValue => 1,
        E::hipErrorOutOfMemory => 2,
        E::hipErrorNotInitialized => 3,
        E::hipErrorDeinitialized => 4,
        E::hipErrorInsufficientDriver => 35,
        E::hipErrorPriorLaunchFailure => 53,
        E::hipErrorNotFound => 500,
        E::hipErrorNotReady => 600,
        E::hipErrorIllegalAddress => 700,
        E::hipErrorLaunchOutOfResources => 701,
        E::hipErrorLaunchTimeOut => 702,
        E::hipErrorLaunchFailure => 719,
        _ => u32::MAX,
    }
}

impl Error {
    /// Build a [`Error::Runtime`] from a raw HIP status code.
    #[must_use]
    pub(crate) fn runtime(code: u32, op: &'static str) -> Self {
        RuntimeSnafu {
            kind: ErrorKind::from_raw(code),
            code,
            op,
        }
        .build()
    }
}

/// Convert a raw `hipError_t` value into `Result<(), Error>`.
///
/// # Errors
///
/// Returns [`Error::Runtime`] with the classified [`ErrorKind`] when
/// the code is anything other than `hipSuccess`.
pub fn check(code: ffi::hipError_t, op: &'static str) -> Result<()> {
    if code == ffi::hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(Error::runtime(hipError_t_code(code), op))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions use expect() directly")]

    use super::*;

    /// Every raw code `hipError_t_from_u32` maps to a named variant.
    /// Single source of truth for the round-trip test below — kept
    /// separate from the match arms so the test fails loudly on a
    /// forward/reverse mismatch instead of trivially agreeing with
    /// itself.
    const MAPPED_CODES: &[u32] = &[0, 1, 2, 3, 4, 35, 53, 500, 600, 700, 701, 702, 719];

    #[test]
    fn mapped_codes_round_trip_through_hip_error_t() {
        for &code in MAPPED_CODES {
            let variant =
                hipError_t_from_u32(code).expect("MAPPED_CODES entry must have a forward mapping");
            assert_eq!(
                hipError_t_code(variant),
                code,
                "code {code} does not round-trip through hipError_t_code \
                 (forward and reverse mappings disagree)"
            );
        }
    }

    #[test]
    fn unmapped_code_falls_through_to_unknown() {
        assert!(hipError_t_from_u32(9_999).is_none());
        assert_eq!(ErrorKind::from_raw(9_999), ErrorKind::Unknown(9_999));
    }
}
