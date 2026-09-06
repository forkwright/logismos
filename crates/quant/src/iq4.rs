//! Shared IQ4 reconstruction facts and fp16-scale admission.

use half::f16;

use crate::error::NonFiniteIq4ScaleSnafu;
use crate::{Result, RowFormat};

pub(crate) const SCALE_BYTES: usize = 2;

// WHY: The operator-approved exception is exactly these finite interoperability
// values. Provenance: ggml-org/llama.cpp@6a1a922d269908a29cbd4b49c27e6a8e7fd10fae,
// ggml/src/ggml-common.h:1120-1122. No upstream expression or decoder is used.
pub(crate) const RECONSTRUCTION_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub(crate) fn finite_scale(format: RowFormat, bytes: [u8; SCALE_BYTES]) -> Result<f32> {
    let bits = u16::from_le_bytes(bytes);
    let scale = f16::from_bits(bits).to_f32();
    if !scale.is_finite() {
        return NonFiniteIq4ScaleSnafu {
            format,
            field: "scale",
            bits,
        }
        .fail();
    }
    Ok(scale)
}
