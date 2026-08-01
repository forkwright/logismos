//! `TurboQuant` block scheme and its derived byte geometry.

use core::fmt;

use crate::{
    TURBO3_0_BITS_PER_INDEX, TURBO3_0_BLOCK_BYTES, TURBO3_0_MAX_INDEX,
    TURBO3_0_PACKED_BYTES_PER_BLOCK, TURBO4_0_BITS_PER_INDEX, TURBO4_0_BLOCK_BYTES,
    TURBO4_0_MAX_INDEX, TURBO4_0_PACKED_BYTES_PER_BLOCK,
};

/// Supported `TurboQuant` block schemes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TurboQuantScheme {
    /// 3-bit index payload with a 2-byte fp16 norm.
    Turbo3_0,
    /// 4-bit index payload with a 2-byte fp16 norm.
    Turbo4_0,
}

impl TurboQuantScheme {
    /// Returns the number of payload bits per quantized index.
    #[must_use]
    pub const fn bits_per_index(self) -> usize {
        match self {
            Self::Turbo3_0 => TURBO3_0_BITS_PER_INDEX,
            Self::Turbo4_0 => TURBO4_0_BITS_PER_INDEX,
        }
    }

    /// Returns the packed payload bytes per 32-value block.
    #[must_use]
    pub const fn packed_bytes_per_block(self) -> usize {
        match self {
            Self::Turbo3_0 => TURBO3_0_PACKED_BYTES_PER_BLOCK,
            Self::Turbo4_0 => TURBO4_0_PACKED_BYTES_PER_BLOCK,
        }
    }

    /// Returns the total bytes per 32-value block, including the norm.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Turbo3_0 => TURBO3_0_BLOCK_BYTES,
            Self::Turbo4_0 => TURBO4_0_BLOCK_BYTES,
        }
    }

    pub(crate) const fn max_index(self) -> u8 {
        match self {
            Self::Turbo3_0 => TURBO3_0_MAX_INDEX,
            Self::Turbo4_0 => TURBO4_0_MAX_INDEX,
        }
    }
}

impl fmt::Display for TurboQuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Turbo3_0 => f.write_str("turbo3_0"),
            Self::Turbo4_0 => f.write_str("turbo4_0"),
        }
    }
}
