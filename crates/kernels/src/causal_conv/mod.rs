//! Bounded CPU reference for dense causal convolution.
//!
//! This module is a model-agnostic primitive: it has no activation, bias,
//! normalization, packing, cache, device, or model-family semantics. Its
//! explicit raw-input history permits callers to compose chunked evaluation
//! without hidden mutable state.

use snafu::{ResultExt, Snafu};

const CAUSAL_CONVOLUTION: &str = "causal_conv_fwd";

/// Result alias for the bounded causal-convolution reference.
pub type CausalConvResult<T> = core::result::Result<T, CausalConvError>;

/// Exact logical `f32` capacities owned by one causal-convolution evaluation.
///
/// WHY: model executors can compose the kernel's checked allocation requests
/// without duplicating its shape arithmetic or mistaking requested capacity for
/// allocator capacity or resident memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalConvAllocationPlan {
    output: usize,
    weights: usize,
    history: usize,
}

impl CausalConvAllocationPlan {
    /// Derive the allocation requests for one admitted convolution shape.
    ///
    /// # Errors
    ///
    /// Returns [`CausalConvError`] when a required dimension is zero or a
    /// declared-shape calculation overflows.
    pub fn try_from_dimensions(
        token_count: usize,
        channel_count: usize,
        width: usize,
    ) -> CausalConvResult<Self> {
        validate_nonzero_dimension("channel_count", channel_count)?;
        validate_nonzero_dimension("width", width)?;
        let history_width = checked_subtract(width, 1, "width - 1")?;
        Ok(Self {
            output: checked_product(token_count, channel_count, "token_count * channel_count")?,
            weights: checked_product(channel_count, width, "channel_count * width")?,
            history: checked_product(channel_count, history_width, "channel_count * (width - 1)")?,
        })
    }

    /// Return the exact requested output capacity.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output
    }

    /// Return the exact requested final-history capacity.
    #[must_use]
    pub const fn history_elements(self) -> usize {
        self.history
    }

    const fn weight_elements(self) -> usize {
        self.weights
    }
}

/// Failures while admitting or evaluating the bounded causal-convolution reference.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum CausalConvError {
    /// A required channel or kernel-width dimension was zero.
    #[snafu(display("{CAUSAL_CONVOLUTION}: {dimension} must be greater than zero"))]
    ZeroDimension {
        /// The rejected dimension.
        dimension: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Multiplying dimensions or positions could not be represented by `usize`.
    #[snafu(display("{CAUSAL_CONVOLUTION}: {dimensions} overflows usize"))]
    DimensionOverflow {
        /// The operation whose result overflowed.
        dimensions: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A dense input did not match its declared shape.
    #[snafu(display(
        "{CAUSAL_CONVOLUTION}: {input} length {actual} does not match expected {expected}"
    ))]
    LengthMismatch {
        /// Input name.
        input: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An admitted scalar was not finite.
    #[snafu(display("{CAUSAL_CONVOLUTION}: {input}[{index}] is not finite"))]
    NonFiniteInput {
        /// Input name.
        input: &'static str,
        /// Flat element index.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An intermediate product, accumulation, or output was not finite.
    #[snafu(display("{CAUSAL_CONVOLUTION}: non-finite value during {stage} at index {index}"))]
    NonFiniteArithmetic {
        /// Named computation stage.
        stage: &'static str,
        /// Flat element index within that stage.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A bounded-by-shape output allocation could not be reserved.
    #[snafu(display(
        "{CAUSAL_CONVOLUTION}: could not reserve {elements} elements for {allocation}"
    ))]
    Allocation {
        /// Allocation role.
        allocation: &'static str,
        /// Requested element count.
        elements: usize,
        /// Allocation failure reported by the standard library.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Validated dense input for a causal convolution.
///
/// The supported layout is input `[T, C]`, weights `[C, W]`, and initial
/// raw-input history `[C, W - 1]`; all are row-major `f32`. Taps and history
/// are ordered oldest-to-newest. `T` may be zero, preserving the history.
#[derive(Debug, Clone, Copy)]
pub struct CausalConvInput<'a> {
    input: &'a [f32],
    weights: &'a [f32],
    history: &'a [f32],
    token_count: usize,
    channel_count: usize,
    width: usize,
    allocations: CausalConvAllocationPlan,
}

impl<'a> CausalConvInput<'a> {
    /// Admit an exact dense causal-convolution input.
    ///
    /// # Errors
    ///
    /// Returns [`CausalConvError`] when a required dimension is zero, a
    /// declared-shape calculation overflows, a buffer length differs from its
    /// declared shape, or a supplied scalar is non-finite.
    pub fn new(
        input: &'a [f32],
        weights: &'a [f32],
        history: &'a [f32],
        token_count: usize,
        channel_count: usize,
        width: usize,
    ) -> CausalConvResult<Self> {
        let allocations =
            CausalConvAllocationPlan::try_from_dimensions(token_count, channel_count, width)?;

        validate_length("input", input.len(), allocations.output_elements())?;
        validate_length("weights", weights.len(), allocations.weight_elements())?;
        validate_length("history", history.len(), allocations.history_elements())?;
        validate_scalars("input", input)?;
        validate_scalars("weights", weights)?;
        validate_scalars("history", history)?;

        Ok(Self {
            input,
            weights,
            history,
            token_count,
            channel_count,
            width,
            allocations,
        })
    }

    fn history_width(&self) -> CausalConvResult<usize> {
        checked_subtract(self.width, 1, "width - 1")
    }

    fn window_sample(&self, channel_index: usize, window_position: usize) -> CausalConvResult<f32> {
        let history_width = self.history_width()?;
        if window_position < history_width {
            let history_start = checked_product(
                channel_index,
                history_width,
                "channel index * history width",
            )?;
            let history_index = checked_add(history_start, window_position, "history index")?;
            return read_scalar(self.history, history_index, "history", self.history.len());
        }

        let token_index = checked_subtract(
            window_position,
            history_width,
            "window position - history width",
        )?;
        let input_start = checked_product(
            token_index,
            self.channel_count,
            "token index * channel count",
        )?;
        let input_index = checked_add(input_start, channel_index, "input index")?;
        read_scalar(self.input, input_index, "input", self.input.len())
    }

    fn weight(&self, channel_index: usize, tap_index: usize) -> CausalConvResult<f32> {
        let weight_start = checked_product(channel_index, self.width, "channel index * width")?;
        let weight_index = checked_add(weight_start, tap_index, "weight index")?;
        read_scalar(self.weights, weight_index, "weights", self.weights.len())
    }
}

/// Output and final raw-input history from [`causal_conv_fwd`].
#[derive(Debug, Clone, PartialEq)]
pub struct CausalConvOutput {
    output: Vec<f32>,
    history: Vec<f32>,
}

impl CausalConvOutput {
    /// Return the dense `[T, C]` output in row-major order.
    #[must_use]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Return the final raw-input `[C, W - 1]` history in row-major order.
    #[must_use]
    pub fn history(&self) -> &[f32] {
        &self.history
    }
}

/// Evaluate the bounded causal-convolution reference.
///
/// For every output `[t, c]`, weights `[c, :]` multiply the oldest-to-newest
/// window ending at input `[t, c]`. The first `W - 1` window positions come
/// from the immutable input history. It performs no activation, bias,
/// normalization, packing, or model-specific interpretation.
///
/// # Errors
///
/// Returns [`CausalConvError::NonFiniteArithmetic`] when an otherwise finite
/// product or accumulation exceeds the `f32` domain, or
/// [`CausalConvError::Allocation`] if an output reservation fails.
pub fn causal_conv_fwd(input: &CausalConvInput<'_>) -> CausalConvResult<CausalConvOutput> {
    let history_width = input.history_width()?;
    let mut output = reserve_f32("output", input.allocations.output_elements())?;

    for token_index in 0..input.token_count {
        for channel_index in 0..input.channel_count {
            let output_index = checked_add(
                checked_product(
                    token_index,
                    input.channel_count,
                    "token index * channel count",
                )?,
                channel_index,
                "output index",
            )?;
            let mut accumulator = 0.0_f32;
            for tap_index in 0..input.width {
                let window_position =
                    checked_add(token_index, tap_index, "token index + tap index")?;
                let sample = input.window_sample(channel_index, window_position)?;
                let product = sample * input.weight(channel_index, tap_index)?;
                ensure_finite(product, "tap product", output_index)?;
                accumulator += product;
                ensure_finite(accumulator, "output accumulation", output_index)?;
            }
            output.push(accumulator);
        }
    }

    let mut final_history = reserve_f32("final history", input.allocations.history_elements())?;
    for channel_index in 0..input.channel_count {
        for history_index in 0..history_width {
            let window_position = checked_add(
                input.token_count,
                history_index,
                "token count + history index",
            )?;
            final_history.push(input.window_sample(channel_index, window_position)?);
        }
    }

    Ok(CausalConvOutput {
        output,
        history: final_history,
    })
}

fn checked_product(left: usize, right: usize, dimensions: &'static str) -> CausalConvResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| DimensionOverflowSnafu { dimensions }.build())
}

fn checked_add(left: usize, right: usize, dimensions: &'static str) -> CausalConvResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| DimensionOverflowSnafu { dimensions }.build())
}

fn checked_subtract(
    left: usize,
    right: usize,
    dimensions: &'static str,
) -> CausalConvResult<usize> {
    left.checked_sub(right)
        .ok_or_else(|| DimensionOverflowSnafu { dimensions }.build())
}

fn validate_nonzero_dimension(dimension: &'static str, value: usize) -> CausalConvResult<()> {
    if value == 0 {
        return ZeroDimensionSnafu { dimension }.fail();
    }
    Ok(())
}

fn validate_length(input: &'static str, actual: usize, expected: usize) -> CausalConvResult<()> {
    if actual != expected {
        return LengthMismatchSnafu {
            input,
            expected,
            actual,
        }
        .fail();
    }
    Ok(())
}

fn validate_scalars(input: &'static str, values: &[f32]) -> CausalConvResult<()> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return NonFiniteInputSnafu { input, index }.fail();
        }
    }
    Ok(())
}

fn read_scalar(
    values: &[f32],
    index: usize,
    input: &'static str,
    expected: usize,
) -> CausalConvResult<f32> {
    values.get(index).copied().ok_or_else(|| {
        LengthMismatchSnafu {
            input,
            expected,
            actual: values.len(),
        }
        .build()
    })
}

fn ensure_finite(value: f32, stage: &'static str, index: usize) -> CausalConvResult<()> {
    if !value.is_finite() {
        return NonFiniteArithmeticSnafu { stage, index }.fail();
    }
    Ok(())
}

fn reserve_f32(allocation: &'static str, elements: usize) -> CausalConvResult<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .context(AllocationSnafu {
            allocation,
            elements,
        })?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHANNEL_COUNT: usize = 2;
    const TOKEN_COUNT: usize = 4;
    const WIDTH: usize = 3;
    const ORACLE_TOLERANCE: f64 = 1e-6;

    #[test]
    fn asymmetric_taps_pin_oldest_to_newest_order() -> CausalConvResult<()> {
        let input = [7.0, 11.0];
        let weights = [2.0, 3.0, 5.0];
        let history = [10.0, 20.0];
        let admitted = CausalConvInput::new(&input, &weights, &history, 2, 1, 3)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert_eq!(
            actual.output(),
            &[115.0, 116.0],
            "tap ordering must be oldest-to-newest"
        );
        assert_eq!(
            actual.history(),
            &[7.0, 11.0],
            "final history must retain newest raw inputs"
        );
        Ok(())
    }

    #[test]
    fn channels_are_independent() -> CausalConvResult<()> {
        let input = [4.0, 8.0, 6.0, 9.0];
        let weights = [2.0, 3.0, 5.0, 7.0];
        let history = [1.0, 10.0];
        let admitted = CausalConvInput::new(&input, &weights, &history, 2, 2, 2)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert_eq!(
            actual.output(),
            &[14.0, 106.0, 26.0, 103.0],
            "channels must use their own taps"
        );
        assert_eq!(
            actual.history(),
            &[6.0, 9.0],
            "each channel must retain its own history"
        );
        Ok(())
    }

    #[test]
    fn width_one_has_no_history() -> CausalConvResult<()> {
        let input = [2.0, 3.0];
        let weights = [4.0];
        let admitted = CausalConvInput::new(&input, &weights, &[], 2, 1, 1)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert_eq!(
            actual.output(),
            &[8.0, 12.0],
            "width-one output must use the current sample"
        );
        assert!(
            actual.history().is_empty(),
            "width one must return no history"
        );
        Ok(())
    }

    #[test]
    fn empty_sequence_preserves_history() -> CausalConvResult<()> {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let history = [10.0, 20.0, 30.0, 40.0];
        let admitted = CausalConvInput::new(&[], &weights, &history, 0, 2, 3)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert!(
            actual.output().is_empty(),
            "an empty sequence must produce no output"
        );
        assert_eq!(
            actual.history(),
            history,
            "an empty sequence must preserve history"
        );
        Ok(())
    }

    #[test]
    fn causal_convolution_matches_independent_f64_oracle() -> CausalConvResult<()> {
        let input = [0.25, -1.0, 1.5, 0.75, -0.5, 2.0, 1.25, -0.25];
        let weights = [0.5, -1.0, 0.25, 1.25, 0.75, -0.5];
        let history = [-2.0, 0.5, 1.0, -1.5];
        let admitted = CausalConvInput::new(
            &input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let actual = causal_conv_fwd(&admitted)?;
        let (expected_output, expected_history) = oracle_causal_conv(
            &input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;

        assert_close_f64(actual.output(), &expected_output, "oracle output");
        assert_close_f64(actual.history(), &expected_history, "oracle history");
        Ok(())
    }

    #[test]
    fn every_chunk_partition_matches_full_evaluation() -> CausalConvResult<()> {
        let input = [0.25, -1.0, 1.5, 0.75, -0.5, 2.0, 1.25, -0.25];
        let weights = [0.5, -1.0, 0.25, 1.25, 0.75, -0.5];
        let history = [-2.0, 0.5, 1.0, -1.5];
        let full_input = CausalConvInput::new(
            &input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let full = causal_conv_fwd(&full_input)?;

        for split_token in 0..=TOKEN_COUNT {
            let first_token_count = split_token;
            let first_input_len = checked_product(
                first_token_count,
                CHANNEL_COUNT,
                "test first token count * channel count",
            )?;
            let first_admitted = CausalConvInput::new(
                input.get(..first_input_len).ok_or_else(|| {
                    LengthMismatchSnafu {
                        input: "test input prefix",
                        expected: first_input_len,
                        actual: input.len(),
                    }
                    .build()
                })?,
                &weights,
                &history,
                first_token_count,
                CHANNEL_COUNT,
                WIDTH,
            )?;
            let first = causal_conv_fwd(&first_admitted)?;
            let remaining_token_count =
                checked_subtract(TOKEN_COUNT, first_token_count, "test remaining token count")?;
            let second_admitted = CausalConvInput::new(
                input.get(first_input_len..).ok_or_else(|| {
                    LengthMismatchSnafu {
                        input: "test input suffix",
                        expected: input.len(),
                        actual: first_input_len,
                    }
                    .build()
                })?,
                &weights,
                first.history(),
                remaining_token_count,
                CHANNEL_COUNT,
                WIDTH,
            )?;
            let second = causal_conv_fwd(&second_admitted)?;
            let mut joined_output = first.output().to_vec();
            joined_output.extend_from_slice(second.output());

            assert_eq!(
                full.output(),
                joined_output,
                "split at token {split_token} must preserve every output"
            );
            assert_eq!(
                full.history(),
                second.history(),
                "split at token {split_token} must preserve final history"
            );
        }
        Ok(())
    }

    #[test]
    fn output_does_not_depend_on_future_input() -> CausalConvResult<()> {
        let prefix = [0.25, -1.0, 1.5, 0.75];
        let first_input = [0.25, -1.0, 1.5, 0.75, -0.5, 2.0, 1.25, -0.25];
        let second_input = [0.25, -1.0, 1.5, 0.75, 99.0, -200.0, 300.0, -400.0];
        let weights = [0.5, -1.0, 0.25, 1.25, 0.75, -0.5];
        let history = [-2.0, 0.5, 1.0, -1.5];
        let first_admitted = CausalConvInput::new(
            &first_input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let second_admitted = CausalConvInput::new(
            &second_input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let first = causal_conv_fwd(&first_admitted)?;
        let second = causal_conv_fwd(&second_admitted)?;
        let prefix_len = prefix.len();

        assert_eq!(
            first.output().get(..prefix_len),
            second.output().get(..prefix_len),
            "future samples must not affect the shared output prefix"
        );
        Ok(())
    }

    #[test]
    fn malformed_nonfinite_overflow_and_arithmetic_inputs_are_rejected() -> CausalConvResult<()> {
        let finite = [1.0_f32; 4];
        let malformed = CausalConvInput::new(&finite[..3], &finite, &finite[..2], 2, 2, 2);
        assert!(
            matches!(
                malformed,
                Err(CausalConvError::LengthMismatch { input: "input", .. })
            ),
            "mismatched input length must fail admission"
        );

        let zero_channel = CausalConvInput::new(&[], &[], &[], 0, 0, 1);
        assert!(
            matches!(
                zero_channel,
                Err(CausalConvError::ZeroDimension {
                    dimension: "channel_count",
                    ..
                })
            ),
            "zero channel count must fail admission"
        );
        let zero_width = CausalConvInput::new(&[], &[], &[], 0, 1, 0);
        assert!(
            matches!(
                zero_width,
                Err(CausalConvError::ZeroDimension {
                    dimension: "width",
                    ..
                })
            ),
            "zero width must fail admission"
        );

        let overflowing_shape = CausalConvInput::new(&[], &[], &[], 0, 2, usize::MAX);
        assert!(
            matches!(
                overflowing_shape,
                Err(CausalConvError::DimensionOverflow {
                    dimensions: "channel_count * width",
                    ..
                })
            ),
            "overflowing shape product must fail admission"
        );

        let nonfinite = CausalConvInput::new(&[1.0], &[f32::NAN], &[], 1, 1, 1);
        assert!(
            matches!(
                nonfinite,
                Err(CausalConvError::NonFiniteInput {
                    input: "weights",
                    ..
                })
            ),
            "non-finite weights must fail admission"
        );
        let nonfinite_input = CausalConvInput::new(&[f32::INFINITY], &[1.0], &[], 1, 1, 1);
        assert!(
            matches!(
                nonfinite_input,
                Err(CausalConvError::NonFiniteInput { input: "input", .. })
            ),
            "non-finite input must fail admission"
        );
        let nonfinite_history =
            CausalConvInput::new(&[1.0], &[1.0, 1.0], &[f32::NEG_INFINITY], 1, 1, 2);
        assert!(
            matches!(
                nonfinite_history,
                Err(CausalConvError::NonFiniteInput {
                    input: "history",
                    ..
                })
            ),
            "non-finite history must fail admission"
        );

        let initial_history = [f32::MAX];
        let arithmetic = CausalConvInput::new(&[1.0], &[f32::MAX, 1.0], &initial_history, 1, 1, 2)?;
        let arithmetic_result = causal_conv_fwd(&arithmetic);
        assert!(
            matches!(
                arithmetic_result,
                Err(CausalConvError::NonFiniteArithmetic {
                    stage: "tap product",
                    ..
                })
            ),
            "non-finite arithmetic must return no partial result"
        );
        assert_eq!(
            initial_history
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![f32::MAX.to_bits()],
            "the immutable caller history must remain bitwise unchanged after rejection"
        );
        Ok(())
    }

    fn oracle_causal_conv(
        input: &[f32],
        weights: &[f32],
        history: &[f32],
        token_count: usize,
        channel_count: usize,
        width: usize,
    ) -> CausalConvResult<(Vec<f64>, Vec<f64>)> {
        let history_width = checked_subtract(width, 1, "oracle width - 1")?;
        let output_len = checked_product(
            token_count,
            channel_count,
            "oracle token count * channel count",
        )?;
        let history_len = checked_product(
            channel_count,
            history_width,
            "oracle channel count * history width",
        )?;
        let weight_len = checked_product(channel_count, width, "oracle channel count * width")?;
        let mut output = Vec::with_capacity(output_len);
        for token_index in 0..token_count {
            for channel_index in 0..channel_count {
                let mut accumulator = 0.0_f64;
                for tap_index in 0..width {
                    let position =
                        checked_add(token_index, tap_index, "oracle token index + tap index")?;
                    let sample = if position < history_width {
                        let index = checked_add(
                            checked_product(
                                channel_index,
                                history_width,
                                "oracle channel * history width",
                            )?,
                            position,
                            "oracle history index",
                        )?;
                        history.get(index).copied().ok_or_else(|| {
                            LengthMismatchSnafu {
                                input: "oracle history",
                                expected: history_len,
                                actual: history.len(),
                            }
                            .build()
                        })?
                    } else {
                        let token = checked_subtract(
                            position,
                            history_width,
                            "oracle position - history width",
                        )?;
                        let index = checked_add(
                            checked_product(token, channel_count, "oracle token * channel count")?,
                            channel_index,
                            "oracle input index",
                        )?;
                        input.get(index).copied().ok_or_else(|| {
                            LengthMismatchSnafu {
                                input: "oracle input",
                                expected: output_len,
                                actual: input.len(),
                            }
                            .build()
                        })?
                    };
                    let weight_index = checked_add(
                        checked_product(channel_index, width, "oracle channel * width")?,
                        tap_index,
                        "oracle weight index",
                    )?;
                    let weight = weights.get(weight_index).copied().ok_or_else(|| {
                        LengthMismatchSnafu {
                            input: "oracle weights",
                            expected: weight_len,
                            actual: weights.len(),
                        }
                        .build()
                    })?;
                    accumulator += f64::from(sample) * f64::from(weight);
                }
                output.push(accumulator);
            }
        }

        let mut final_history = Vec::with_capacity(history_len);
        for channel_index in 0..channel_count {
            for history_index in 0..history_width {
                let position = checked_add(
                    token_count,
                    history_index,
                    "oracle token count + history index",
                )?;
                let sample = if position < history_width {
                    let index = checked_add(
                        checked_product(
                            channel_index,
                            history_width,
                            "oracle channel * history width",
                        )?,
                        position,
                        "oracle history index",
                    )?;
                    history.get(index).copied().ok_or_else(|| {
                        LengthMismatchSnafu {
                            input: "oracle history",
                            expected: history_len,
                            actual: history.len(),
                        }
                        .build()
                    })?
                } else {
                    let token = checked_subtract(
                        position,
                        history_width,
                        "oracle position - history width",
                    )?;
                    let index = checked_add(
                        checked_product(token, channel_count, "oracle token * channel count")?,
                        channel_index,
                        "oracle input index",
                    )?;
                    input.get(index).copied().ok_or_else(|| {
                        LengthMismatchSnafu {
                            input: "oracle input",
                            expected: output_len,
                            actual: input.len(),
                        }
                        .build()
                    })?
                };
                final_history.push(f64::from(sample));
            }
        }
        Ok((output, final_history))
    }

    fn assert_close_f64(actual: &[f32], expected: &[f64], name: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{name} length must match oracle"
        );
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            let difference = (f64::from(*actual_value) - *expected_value).abs();
            assert!(
                difference <= ORACLE_TOLERANCE,
                "{name}[{index}] differs from independent oracle by {difference}"
            );
        }
    }
}
