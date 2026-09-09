//! Shared Qwen3.5 text MRoPE coefficient derivation.

use crate::Result;
use crate::error::{ArithmeticOverflowSnafu, ExecutionArithmeticSnafu};
use num_traits::ToPrimitive;

/// Validated text MRoPE parameters reused by CPU and native execution.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TextMrope {
    rotary_width: usize,
    rope_base: f64,
    sections: [usize; 4],
}

impl TextMrope {
    pub(crate) const fn new(rotary_width: usize, rope_base: f64, sections: [usize; 4]) -> Self {
        Self {
            rotary_width,
            rope_base,
            sections,
        }
    }

    pub(crate) const fn rotary_width(self) -> usize {
        self.rotary_width
    }

    fn axis(self, pair: usize) -> Result<usize> {
        let total = self.sections.iter().try_fold(0_usize, |sum, section| {
            sum.checked_add(*section).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "text MRoPE section total",
                }
                .build()
            })
        })?;
        let sector = pair % total;
        if sector % 3 == 1 && sector < 3 * self.sections[1] {
            return Ok(1);
        }
        if sector % 3 == 2 && sector < 3 * self.sections[2] {
            return Ok(2);
        }
        if sector.is_multiple_of(3) && sector < 3 * self.sections[0] {
            return Ok(0);
        }
        Ok(3)
    }
}

/// Derive one f32 cosine/sine pair for a half-split rotary coordinate.
pub(crate) fn text_mrope_coefficient(
    mrope: TextMrope,
    position: usize,
    pair: usize,
) -> Result<(f32, f32)> {
    let position = f64::from(i32::try_from(position).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "text MRoPE position",
        }
        .build()
    })?);
    let rotary_width = mrope.rotary_width().to_f64().ok_or_else(|| {
        ArithmeticOverflowSnafu {
            context: "text MRoPE rotary width",
        }
        .build()
    })?;
    let pair_f64 = pair.to_f64().ok_or_else(|| {
        ArithmeticOverflowSnafu {
            context: "text MRoPE pair",
        }
        .build()
    })?;
    let exponent = (2.0 * pair_f64) / rotary_width;
    let axis = mrope.axis(pair)?;
    let text_position = if axis == 3 { 0.0 } else { position };
    let angle = text_position / mrope.rope_base.powf(exponent);
    let cosine = angle.cos().to_f32().ok_or_else(|| {
        ExecutionArithmeticSnafu {
            stage: "text MRoPE cosine",
            index: pair,
        }
        .build()
    })?;
    let sine = angle.sin().to_f32().ok_or_else(|| {
        ExecutionArithmeticSnafu {
            stage: "text MRoPE sine",
            index: pair,
        }
        .build()
    })?;
    Ok((cosine, sine))
}

#[cfg(test)]
mod tests {
    use super::{TextMrope, text_mrope_coefficient};

    const POSITION: usize = 3;
    const ROPE_BASE: f64 = 10_000.0;
    const COEFFICIENT_EPSILON: f32 = 1.0e-6;

    #[test]
    fn coefficients_match_independent_text_position_angles() -> std::result::Result<(), String> {
        let position = POSITION as f64;
        let expected = [position, position / ROPE_BASE.sqrt()];
        for (index, angle) in expected.into_iter().enumerate() {
            let (actual_cosine, actual_sine) =
                text_mrope_coefficient(TextMrope::new(4, ROPE_BASE, [1, 1, 0, 0]), POSITION, index)
                    .map_err(|error| error.to_string())?;
            let expected_cosine = angle.cos() as f32;
            let expected_sine = angle.sin() as f32;
            assert!(
                (actual_cosine - expected_cosine).abs() <= COEFFICIENT_EPSILON,
                "cosine coefficient must derive from the independent text-position angle"
            );
            assert!(
                (actual_sine - expected_sine).abs() <= COEFFICIENT_EPSILON,
                "sine coefficient must derive from the independent text-position angle"
            );
        }
        Ok(())
    }
}
