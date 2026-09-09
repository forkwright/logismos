//! Authoritative native numerical-status representation.

/// Device representation byte count for one sticky native numerical status.
pub const NATIVE_NUMERICAL_STATUS_BYTES: usize = core::mem::size_of::<u32>();

/// Sticky failure categories reported by checked native arithmetic.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[cfg(test)]
const F32_EXPONENT_MASK: u32 = 0x7f80_0000;
#[cfg(test)]
const F32_SIGNIFICAND_MASK: u32 = 0x007f_ffff;
#[cfg(test)]
const F16_EXPONENT_MASK: u16 = 0x7c00;
#[cfg(test)]
const F16_SIGNIFICAND_MASK: u16 = 0x03ff;

#[cfg(test)]
const fn f32_bits_are_subnormal(bits: u32) -> bool {
    bits & F32_EXPONENT_MASK == 0 && bits & F32_SIGNIFICAND_MASK != 0
}

#[cfg(test)]
const fn f32_bits_are_nonfinite(bits: u32) -> bool {
    bits & F32_EXPONENT_MASK == F32_EXPONENT_MASK
}

#[cfg(test)]
const fn f16_bits_are_subnormal(bits: u16) -> bool {
    bits & F16_EXPONENT_MASK == 0 && bits & F16_SIGNIFICAND_MASK != 0
}

#[cfg(test)]
const fn f16_bits_are_nonfinite(bits: u16) -> bool {
    bits & F16_EXPONENT_MASK == F16_EXPONENT_MASK
}

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

#[cfg(test)]
mod tests {
    use super::{
        NATIVE_NUMERICAL_STATUS_KNOWN_BITS, NativeNumericalStatusCategory,
        NativeNumericalStatusMask, f16_bits_are_nonfinite, f16_bits_are_subnormal,
        f32_bits_are_nonfinite, f32_bits_are_subnormal,
    };

    #[test]
    fn mask_accepts_combined_typed_categories() {
        let bits = NativeNumericalStatusCategory::InputSubnormal.bit()
            | NativeNumericalStatusCategory::ArithmeticNonFinite.bit();
        let Some(mask) = NativeNumericalStatusMask::from_bits(bits) else {
            panic!("known combined numerical-status bits must construct a typed mask");
        };

        assert!(mask.contains(NativeNumericalStatusCategory::InputSubnormal));
        assert!(mask.contains(NativeNumericalStatusCategory::ArithmeticNonFinite));
        assert!(!mask.contains(NativeNumericalStatusCategory::InputNonFinite));
    }

    #[test]
    fn mask_refuses_unknown_bits() {
        const UNKNOWN_BIT: u32 = 16;
        let unknown = NATIVE_NUMERICAL_STATUS_KNOWN_BITS | UNKNOWN_BIT;

        assert!(NativeNumericalStatusMask::from_bits(unknown).is_none());
    }

    #[test]
    fn bitwise_float_classification_keeps_subnormals_distinct_from_zero() {
        const F32_ZERO: u32 = 0;
        const F32_SMALLEST_SUBNORMAL: u32 = 1;
        const F32_INFINITY: u32 = 0x7f80_0000;
        const F32_QUIET_NAN: u32 = 0x7fc0_0000;
        const F16_ZERO: u16 = 0;
        const F16_SMALLEST_SUBNORMAL: u16 = 1;
        const F16_INFINITY: u16 = 0x7c00;
        const F16_QUIET_NAN: u16 = 0x7e00;

        assert!(!f32_bits_are_subnormal(F32_ZERO));
        assert!(f32_bits_are_subnormal(F32_SMALLEST_SUBNORMAL));
        assert!(f32_bits_are_nonfinite(F32_INFINITY));
        assert!(f32_bits_are_nonfinite(F32_QUIET_NAN));
        assert!(!f16_bits_are_subnormal(F16_ZERO));
        assert!(f16_bits_are_subnormal(F16_SMALLEST_SUBNORMAL));
        assert!(f16_bits_are_nonfinite(F16_INFINITY));
        assert!(f16_bits_are_nonfinite(F16_QUIET_NAN));
    }
}
