//! Format identities and checked row geometry for executable weight storage.

use core::fmt;

/// Executable serialized row formats supported by `quant`.
///
/// This is deliberately narrower than GGML's storage tag set. Callers must
/// refuse a known-but-not-executable format rather than selecting an
/// approximation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RowFormat {
    /// Little-endian IEEE-754 f32 scalar values.
    F32,
    /// GGML `Q8_0` blocks.
    Q8_0,
    /// GGML `Q4_K` blocks.
    Q4K,
    /// GGML `Q5_K` blocks.
    Q5K,
    /// GGML `Q6_K` blocks.
    Q6K,
    /// GGML `IQ4_NL` blocks.
    IQ4NL,
    /// GGML `IQ4_XS` blocks.
    IQ4XS,
}

impl fmt::Display for RowFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::IQ4NL => "iq4_nl",
            Self::IQ4XS => "iq4_xs",
        };
        formatter.write_str(name)
    }
}
