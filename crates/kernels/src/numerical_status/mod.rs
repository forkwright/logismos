//! Owned sticky numerical-status allocation for checked native arithmetic.

mod codes;

pub use codes::{
    NATIVE_NUMERICAL_STATUS_BYTES, NATIVE_NUMERICAL_STATUS_KNOWN_BITS,
    NativeNumericalStatusCategory, NativeNumericalStatusMask,
};

#[cfg(feature = "gpu")]
use hipcore::{Device, DeviceBuffer};
#[cfg(feature = "gpu")]
use snafu::Snafu;

#[cfg(feature = "gpu")]
use crate::error::{NumericalStatusSnafu, Result};

/// Typed failure reported after reading a synchronized native status allocation.
#[cfg(feature = "gpu")]
#[derive(Debug, Snafu)]
#[non_exhaustive]
pub enum NativeNumericalStatusError {
    /// The device wrote a bit outside the generated native-status contract.
    #[snafu(display("native numerical status contains unknown bits {bits:#010x}"))]
    UnknownBits {
        /// Unrecognized raw device bits.
        bits: u32,
    },

    /// Checked arithmetic observed one or more typed numerical-domain violations.
    #[snafu(display("native numerical status observed typed bits {mask:?}"))]
    Observed {
        /// Full host-validated sticky status mask.
        mask: NativeNumericalStatusMask,
    },

    /// Reading the synchronized status allocation failed.
    #[snafu(transparent)]
    Hip {
        /// Source HIP failure.
        source: hipcore::Error,
    },
}

/// One session-owned sticky device status for checked native arithmetic.
///
/// A status allocation begins at zero and is never reset. A nonzero status
/// makes its enclosing session unusable, so reusing it cannot erase evidence
/// from a submitted operation.
#[cfg(feature = "gpu")]
pub struct NativeNumericalStatus {
    bits: DeviceBuffer<u32>,
}

#[cfg(feature = "gpu")]
impl NativeNumericalStatus {
    /// Allocate one initialized sticky native-status word on `device`.
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            bits: DeviceBuffer::from_host(device, &[0]),
        })
    }

    /// Return the exact device allocation demand for one sticky status word.
    #[must_use]
    pub const fn byte_demand() -> usize {
        NATIVE_NUMERICAL_STATUS_BYTES
    }

    /// Return the opaque device pointer for a checked kernel-launch shim.
    ///
    /// # Safety
    ///
    /// The caller must retain this status owner until the launched work has
    /// completed and may only give the pointer to a checked native launcher.
    #[must_use]
    pub(crate) unsafe fn as_device_ptr(&self) -> *mut u32 {
        self.bits.as_device_ptr()
    }

    /// Read and validate this status after the owning stream has synchronized.
    ///
    /// This only classifies explicit checked-operation operands and results;
    /// it does not establish math-library internals, emitted denorm mode, or
    /// physical-device floating-point qualification.
    pub fn read_after_synchronization(&self) -> Result<()> {
        let mut raw = [0_u32];
        self.bits.copy_to_host(&mut raw).map_err(|source| {
            NumericalStatusSnafu {
                source: NativeNumericalStatusError::Hip { source },
            }
            .build()
        })?;
        let Some(mask) = NativeNumericalStatusMask::from_bits(raw[0]) else {
            return NumericalStatusSnafu {
                source: NativeNumericalStatusError::UnknownBits { bits: raw[0] },
            }
            .fail();
        };
        if mask.is_empty() {
            return Ok(());
        }
        NumericalStatusSnafu {
            source: NativeNumericalStatusError::Observed { mask },
        }
        .fail()
    }
}
