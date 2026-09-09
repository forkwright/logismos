//! Owned sticky numerical-status allocation for checked native arithmetic.

mod codes;

pub use codes::{
    NATIVE_NUMERICAL_STATUS_BYTES, NATIVE_NUMERICAL_STATUS_KNOWN_BITS,
    NativeNumericalStatusCategory,
};

#[cfg(feature = "gpu")]
use hipcore::{Device, DeviceBuffer};
#[cfg(feature = "gpu")]
use snafu::{ResultExt, Snafu};

#[cfg(feature = "gpu")]
use crate::error::{NumericalStatusSnafu, Result};

/// Typed, host-validated sticky native numerical-status mask.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeNumericalStatusMask(u32);

impl NativeNumericalStatusMask {
    /// Construct a typed mask only when every raw bit is defined.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !NATIVE_NUMERICAL_STATUS_KNOWN_BITS == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Return the status representation's raw device bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether no classified numerical failure was recorded.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether this mask includes one typed category.
    #[must_use]
    pub const fn contains(self, category: NativeNumericalStatusCategory) -> bool {
        self.0 & category.bit() != 0
    }
}

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

/// Caller-owned native-status storage with validated geometry and device identity.
///
/// This carrier does not claim that its status word has been initialized. A
/// construction transaction must explicitly initialize the retained owner
/// before it becomes a [`NativeNumericalStatus`].
#[cfg(feature = "gpu")]
pub struct NativeNumericalStatusBuffer {
    bits: DeviceBuffer<u32>,
}

#[cfg(feature = "gpu")]
impl NativeNumericalStatusBuffer {
    /// Initialize the retained status word and consume it into a checked owner.
    ///
    /// The host-to-device copy is synchronous. On failure, the returned error
    /// retains the original allocation with unspecified contents; no initialized
    /// status owner is published.
    ///
    /// # Errors
    ///
    /// Returns [`NativeNumericalStatusInitializationError`] when HIP cannot
    /// complete the zero initialization.
    pub fn initialize(
        self,
    ) -> core::result::Result<NativeNumericalStatus, NativeNumericalStatusInitializationError> {
        let mut bits = self.bits;
        match bits.copy_from_host(&[0]) {
            Ok(()) => Ok(NativeNumericalStatus { bits }),
            Err(source) => Err(NativeNumericalStatusInitializationError { source, bits }),
        }
    }

    /// Consume this carrier and return its original typed device-buffer owner.
    #[must_use]
    pub fn into_buffer(self) -> DeviceBuffer<u32> {
        self.bits
    }
}

/// Failed initialization of a caller-owned native status buffer.
///
/// This failure retains the original allocation with unspecified contents.
/// Native construction must move it into inert teardown custody before
/// abandoning the failure.
#[cfg(feature = "gpu")]
pub struct NativeNumericalStatusInitializationError {
    source: hipcore::Error,
    bits: DeviceBuffer<u32>,
}

#[cfg(feature = "gpu")]
impl core::fmt::Debug for NativeNumericalStatusInitializationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("NativeNumericalStatusInitializationError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "gpu")]
impl core::fmt::Display for NativeNumericalStatusInitializationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.source.fmt(formatter)
    }
}

#[cfg(feature = "gpu")]
impl std::error::Error for NativeNumericalStatusInitializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(feature = "gpu")]
impl NativeNumericalStatusInitializationError {
    /// Borrow the HIP failure that prevented initialization.
    #[must_use]
    pub const fn error(&self) -> &hipcore::Error {
        &self.source
    }

    /// Consume the failure and recover its source and original allocation.
    #[must_use]
    pub fn into_parts(self) -> (hipcore::Error, DeviceBuffer<u32>) {
        (self.source, self.bits)
    }
}

/// Reason a caller-owned numerical-status buffer could not become a status owner.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeNumericalStatusBufferErrorKind {
    /// The buffer belongs to a different process-visible HIP device.
    DeviceMismatch,
    /// The buffer does not contain exactly one native-status word.
    LengthMismatch,
}

/// Rejected caller-owned numerical-status buffer retaining its original owner.
#[cfg(feature = "gpu")]
pub struct NativeNumericalStatusBufferError {
    kind: NativeNumericalStatusBufferErrorKind,
    buffer: DeviceBuffer<u32>,
}

#[cfg(feature = "gpu")]
impl NativeNumericalStatusBufferError {
    /// Return the ownership-preserving rejection category.
    #[must_use]
    pub const fn kind(&self) -> NativeNumericalStatusBufferErrorKind {
        self.kind
    }

    /// Consume this rejection and recover its original typed buffer owner.
    #[must_use]
    pub fn into_buffer(self) -> DeviceBuffer<u32> {
        self.buffer
    }

    /// Consume this rejection and recover its category and original buffer.
    #[must_use]
    pub fn into_parts(self) -> (NativeNumericalStatusBufferErrorKind, DeviceBuffer<u32>) {
        (self.kind, self.buffer)
    }
}

#[cfg(feature = "gpu")]
impl NativeNumericalStatus {
    /// Allocate one initialized sticky native-status word on `device`.
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            bits: DeviceBuffer::from_host(device, &[0])?,
        })
    }

    /// Validate one caller-owned raw status buffer without initializing it.
    ///
    /// This performs no allocation or device write. A rejected buffer remains
    /// owned by [`NativeNumericalStatusBufferError`] for explicit construction
    /// teardown; its device contents and completion state are not interpreted.
    pub fn try_bind_buffer(
        device: &Device,
        bits: DeviceBuffer<u32>,
    ) -> core::result::Result<NativeNumericalStatusBuffer, NativeNumericalStatusBufferError> {
        if bits.len() != 1 {
            return Err(NativeNumericalStatusBufferError {
                kind: NativeNumericalStatusBufferErrorKind::LengthMismatch,
                buffer: bits,
            });
        }
        if bits.device().ordinal() != device.ordinal() {
            return Err(NativeNumericalStatusBufferError {
                kind: NativeNumericalStatusBufferErrorKind::DeviceMismatch,
                buffer: bits,
            });
        }
        Ok(NativeNumericalStatusBuffer { bits })
    }

    /// Consume this status and return its original typed device-buffer owner.
    ///
    /// This performs no synchronization, HIP release, or eviction
    /// acknowledgement. A caller handling submitted work must retain the
    /// returned owner in a quiescing teardown path before it can drop.
    #[must_use]
    pub fn into_buffer(self) -> DeviceBuffer<u32> {
        self.bits
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
        self.bits
            .copy_to_host(&mut raw)
            .map_err(|source| NativeNumericalStatusError::Hip { source })
            .context(NumericalStatusSnafu)?;
        let Some(mask) = NativeNumericalStatusMask::from_bits(raw[0]) else {
            return Err(NativeNumericalStatusError::UnknownBits { bits: raw[0] })
                .context(NumericalStatusSnafu);
        };
        if mask.is_empty() {
            return Ok(());
        }
        Err(NativeNumericalStatusError::Observed { mask }).context(NumericalStatusSnafu)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NATIVE_NUMERICAL_STATUS_KNOWN_BITS, NativeNumericalStatusCategory,
        NativeNumericalStatusMask,
    };

    #[cfg(feature = "gpu")]
    use super::NativeNumericalStatus;

    #[cfg(feature = "gpu")]
    #[test]
    fn native_status_demand_is_one_u32_without_opening_a_device() {
        assert_eq!(
            NativeNumericalStatus::byte_demand(),
            core::mem::size_of::<u32>()
        );
    }

    #[test]
    fn mask_accepts_combined_typed_categories()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let bits = NativeNumericalStatusCategory::InputSubnormal.bit()
            | NativeNumericalStatusCategory::ArithmeticNonFinite.bit();
        let mask = NativeNumericalStatusMask::from_bits(bits)
            .ok_or("known combined numerical-status bits must construct a typed mask")?;

        assert!(mask.contains(NativeNumericalStatusCategory::InputSubnormal));
        assert!(mask.contains(NativeNumericalStatusCategory::ArithmeticNonFinite));
        assert!(!mask.contains(NativeNumericalStatusCategory::InputNonFinite));
        Ok(())
    }

    #[test]
    fn mask_refuses_unknown_bits() {
        const UNKNOWN_BIT: u32 = 16;
        let unknown = NATIVE_NUMERICAL_STATUS_KNOWN_BITS | UNKNOWN_BIT;

        assert!(NativeNumericalStatusMask::from_bits(unknown).is_none());
    }

    #[test]
    fn bitwise_float_classification_keeps_subnormals_distinct_from_zero() {
        const F32_EXPONENT_MASK: u32 = 0x7f80_0000;
        const F32_SIGNIFICAND_MASK: u32 = 0x007f_ffff;
        const F32_ZERO: u32 = 0;
        const F32_SMALLEST_SUBNORMAL: u32 = 1;
        const F32_INFINITY: u32 = F32_EXPONENT_MASK;
        const F32_QUIET_NAN: u32 = F32_EXPONENT_MASK | 0x0040_0000;
        const F16_EXPONENT_MASK: u16 = 0x7c00;
        const F16_SIGNIFICAND_MASK: u16 = 0x03ff;
        const F16_ZERO: u16 = 0;
        const F16_SMALLEST_SUBNORMAL: u16 = 1;
        const F16_INFINITY: u16 = F16_EXPONENT_MASK;
        const F16_QUIET_NAN: u16 = F16_EXPONENT_MASK | 0x0200;

        assert!(!is_subnormal(
            F32_ZERO,
            F32_EXPONENT_MASK,
            F32_SIGNIFICAND_MASK
        ));
        assert!(is_subnormal(
            F32_SMALLEST_SUBNORMAL,
            F32_EXPONENT_MASK,
            F32_SIGNIFICAND_MASK
        ));
        assert!(is_nonfinite(F32_INFINITY, F32_EXPONENT_MASK));
        assert!(is_nonfinite(F32_QUIET_NAN, F32_EXPONENT_MASK));
        assert!(!is_subnormal(
            F16_ZERO,
            F16_EXPONENT_MASK,
            F16_SIGNIFICAND_MASK
        ));
        assert!(is_subnormal(
            F16_SMALLEST_SUBNORMAL,
            F16_EXPONENT_MASK,
            F16_SIGNIFICAND_MASK
        ));
        assert!(is_nonfinite(F16_INFINITY, F16_EXPONENT_MASK));
        assert!(is_nonfinite(F16_QUIET_NAN, F16_EXPONENT_MASK));
    }

    fn is_subnormal<Value>(value: Value, exponent_mask: Value, significand_mask: Value) -> bool
    where
        Value: core::ops::BitAnd<Output = Value> + From<u8> + PartialEq + Copy,
    {
        value & exponent_mask == Value::from(0) && value & significand_mask != Value::from(0)
    }

    fn is_nonfinite<Value>(value: Value, exponent_mask: Value) -> bool
    where
        Value: core::ops::BitAnd<Output = Value> + PartialEq + Copy,
    {
        value & exponent_mask == exponent_mask
    }
}
