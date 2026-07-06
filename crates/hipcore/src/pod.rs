//! The [`BytePod`] marker trait.
//!
//! Crate-private unsafe trait guaranteeing that a `T` is `Copy`, has a
//! well-defined bit layout, and that every bit pattern is a valid
//! inhabitant — i.e. the type can be written to / read from raw memory
//! (device or pinned host) without introducing UB.
//!
//! `bool` is deliberately **not** `BytePod` (a byte value of `2` is not
//! a valid `bool`).

use core::mem::size_of;

/// Marker trait for types safe to memcpy to / from device memory.
///
/// # Safety
///
/// Implementors must guarantee:
/// 1. `size_of::<Self>() > 0` and matches the expected on-device
///    element size.
/// 2. Every bit pattern of length `size_of::<Self>()` is a valid
///    inhabitant of `Self` (no niche optimisations, no `NonZero*`).
/// 3. `Self: Copy`.
///
/// Violating any of these invariants can cause undefined behaviour the
/// next time a [`crate::DeviceBuffer`] is read back or interpreted.
pub unsafe trait BytePod: Copy + 'static {
    /// Size of `Self` in bytes.
    fn size_of_self() -> usize {
        size_of::<Self>()
    }
}

// SAFETY: integer and IEEE float types are `Copy`, have stable bit
// layouts, and every bit pattern is a valid inhabitant.
unsafe impl BytePod for u8 {}
// SAFETY: see above.
unsafe impl BytePod for i8 {}
// SAFETY: see above.
unsafe impl BytePod for u16 {}
// SAFETY: see above.
unsafe impl BytePod for i16 {}
// SAFETY: see above.
unsafe impl BytePod for u32 {}
// SAFETY: see above.
unsafe impl BytePod for i32 {}
// SAFETY: see above.
unsafe impl BytePod for u64 {}
// SAFETY: see above.
unsafe impl BytePod for i64 {}
// SAFETY: every f32 bit pattern is a valid float (NaN included).
unsafe impl BytePod for f32 {}
// SAFETY: every f64 bit pattern is a valid float.
unsafe impl BytePod for f64 {}
// SAFETY: `half::f16` is `#[repr(transparent)] struct f16(u16)`; every
// bit pattern is a valid inhabitant.
unsafe impl BytePod for half::f16 {}
// SAFETY: same argument for `bf16`.
unsafe impl BytePod for half::bf16 {}
