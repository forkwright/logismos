//! Authoritative native numerical-status representation.

/// Device representation byte count for one sticky native numerical status.
pub const NATIVE_NUMERICAL_STATUS_BYTES: usize = core::mem::size_of::<u32>();

/// Sticky failure categories reported by checked native arithmetic.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeNumericalStatusCategory {
    /// An original model operand was subnormal before its first transform.
    InputSubnormal = 1,
    /// An original model operand was infinite or NaN before its first transform.
    InputNonFinite = 2,
    /// An explicit arithmetic operand or result was subnormal.
    ArithmeticSubnormal = 4,
    /// An explicit arithmetic operand or result was infinite or NaN.
    ArithmeticNonFinite = 8,
}

impl NativeNumericalStatusCategory {
    /// Return this category's sticky device bit.
    #[must_use]
    pub const fn bit(self) -> u32 {
        self as u32
    }
}

/// All status bits recognized by this native numerical-status contract.
pub const NATIVE_NUMERICAL_STATUS_KNOWN_BITS: u32 = NativeNumericalStatusCategory::InputSubnormal
    .bit()
    | NativeNumericalStatusCategory::InputNonFinite.bit()
    | NativeNumericalStatusCategory::ArithmeticSubnormal.bit()
    | NativeNumericalStatusCategory::ArithmeticNonFinite.bit();
