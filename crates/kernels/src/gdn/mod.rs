//! Bounded CPU reference for one-head Gated Delta Rule recurrence.
//!
//! This module deliberately accepts only dense one-head recurrent input. It
//! is a correctness oracle for a future device kernel, not a model adapter or
//! a permissive fallback for unsupported GDN variants.
//! Bounds describe the admitted shapes and numerical domain, not a memory
//! quota; allocation exhaustion remains a process-level failure.

use snafu::Snafu;

const GDN_RECURRENCE: &str = "gdn_recurrent_fwd";

/// Result alias for the bounded GDN reference.
pub type GdnResult<T> = core::result::Result<T, GdnError>;

/// Failures while admitting or evaluating the bounded GDN reference.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum GdnError {
    /// A required recurrence dimension was zero.
    #[snafu(display("{GDN_RECURRENCE}: {dimension} must be greater than zero"))]
    ZeroDimension {
        /// The rejected dimension.
        dimension: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Multiplying dimensions could not be represented by `usize`.
    #[snafu(display("{GDN_RECURRENCE}: {dimensions} element count overflows usize"))]
    DimensionProductOverflow {
        /// The multiplied dimensions.
        dimensions: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A dense input did not match its declared shape.
    #[snafu(display(
        "{GDN_RECURRENCE}: {input} length {actual} does not match expected {expected}"
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

    /// An input scalar was not finite.
    #[snafu(display("{GDN_RECURRENCE}: {input}[{index}] is not finite"))]
    NonFiniteInput {
        /// Input name.
        input: &'static str,
        /// Flat element index.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A recurrence intermediate or result was not finite.
    #[snafu(display("{GDN_RECURRENCE}: non-finite value during {stage} at index {index}"))]
    NonFiniteArithmetic {
        /// Named recurrence stage.
        stage: &'static str,
        /// Index within the named stage's token, state or value axis.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Validated dense input for a one-head GDN recurrence.
///
/// The supported domain is `q/k: [T, K]`, `v: [T, V]`, scalar
/// `beta/g: [T]`, and `state: [K, V]`, all `f32`. `g` is natural-log decay;
/// the recurrence applies `exp(g)` before the delta update.
#[derive(Debug, Clone, Copy)]
pub struct RecurrentInput<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    beta: &'a [f32],
    g: &'a [f32],
    scale: f32,
    state: &'a [f32],
    token_count: usize,
    key_dim: usize,
    value_dim: usize,
}

impl<'a> RecurrentInput<'a> {
    /// Admit an exact dense one-head recurrence input.
    ///
    /// # Errors
    ///
    /// Returns [`GdnError`] when a dimension is zero, a shape product
    /// overflows, a buffer length differs from the declared shape, or any
    /// supplied scalar is non-finite.
    #[expect(
        clippy::too_many_arguments,
        reason = "the six input buffers, scale and two dimensions form the fixed recurrence contract"
    )]
    pub fn new(
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        beta: &'a [f32],
        g: &'a [f32],
        scale: f32,
        state: &'a [f32],
        key_dim: usize,
        value_dim: usize,
    ) -> GdnResult<Self> {
        if key_dim == 0 {
            return ZeroDimensionSnafu {
                dimension: "key_dim",
            }
            .fail();
        }
        if value_dim == 0 {
            return ZeroDimensionSnafu {
                dimension: "value_dim",
            }
            .fail();
        }

        let token_count = beta.len();
        if token_count == 0 {
            return ZeroDimensionSnafu {
                dimension: "token_count",
            }
            .fail();
        }

        let query_and_key_len = checked_product(token_count, key_dim, "token_count * key_dim")?;
        let value_len = checked_product(token_count, value_dim, "token_count * value_dim")?;
        let state_len = checked_product(key_dim, value_dim, "key_dim * value_dim")?;

        validate_length("q", q.len(), query_and_key_len)?;
        validate_length("k", k.len(), query_and_key_len)?;
        validate_length("v", v.len(), value_len)?;
        validate_length("g", g.len(), token_count)?;
        validate_length("state", state.len(), state_len)?;
        validate_scalars("q", q)?;
        validate_scalars("k", k)?;
        validate_scalars("v", v)?;
        validate_scalars("beta", beta)?;
        validate_scalars("g", g)?;
        validate_scalars("state", state)?;
        if !scale.is_finite() {
            return NonFiniteInputSnafu {
                input: "scale",
                index: 0_usize,
            }
            .fail();
        }

        Ok(Self {
            q,
            k,
            v,
            beta,
            g,
            scale,
            state,
            token_count,
            key_dim,
            value_dim,
        })
    }
}

/// Output and final state from [`recurrent_fwd`].
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentOutput {
    output: Vec<f32>,
    state: Vec<f32>,
}

impl RecurrentOutput {
    /// Return the dense `[T, V]` output in row-major order.
    #[must_use]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Return the final `[K, V]` state in row-major order.
    #[must_use]
    pub fn state(&self) -> &[f32] {
        &self.state
    }
}

/// Evaluate the bounded one-head recurrent GDN reference.
///
/// It performs, for each token, `state *= exp(g)`,
/// `delta = beta * (v - state^T * k)`, `state += outer(k, delta)`, and
/// `output = state^T * q * scale`. Every intermediate is checked for
/// finiteness; an error returns no partial output or state.
///
/// # Errors
///
/// Returns [`GdnError::NonFiniteArithmetic`] if a decay, projection, delta,
/// state update, or output calculation becomes non-finite.
pub fn recurrent_fwd(input: &RecurrentInput<'_>) -> GdnResult<RecurrentOutput> {
    let output_len = checked_product(
        input.token_count,
        input.value_dim,
        "token_count * value_dim",
    )?;
    let mut state = input.state.to_vec();
    let mut output = vec![0.0_f32; output_len];

    for token_index in 0..input.token_count {
        let q_start = checked_product(token_index, input.key_dim, "token index * key_dim")?;
        let q_end = checked_add(q_start, input.key_dim, "query row end")?;
        let v_start = checked_product(token_index, input.value_dim, "token index * value_dim")?;
        let v_end = checked_add(v_start, input.value_dim, "value row end")?;
        let q_row = read_row(input.q, q_start, q_end, "q")?;
        let k_row = read_row(input.k, q_start, q_end, "k")?;
        let v_row = read_row(input.v, v_start, v_end, "v")?;
        let beta = read_scalar(input.beta, token_index, "beta", input.token_count)?;
        let gate = read_scalar(input.g, token_index, "g", input.token_count)?;

        let decay = gate.exp();
        ensure_finite(decay, "decay", token_index)?;
        for (state_index, state_value) in state.iter_mut().enumerate() {
            *state_value *= decay;
            ensure_finite(*state_value, "state decay", state_index)?;
        }

        let mut state_times_key = vec![0.0_f32; input.value_dim];
        for (key_index, key_value) in k_row.iter().copied().enumerate() {
            for (value_index, accumulator) in state_times_key.iter_mut().enumerate() {
                let state_index = matrix_index(key_index, value_index, input.value_dim)?;
                let state_value = read_scalar(&state, state_index, "state", state.len())?;
                *accumulator += state_value * key_value;
                ensure_finite(*accumulator, "state times key", value_index)?;
            }
        }

        let mut delta = Vec::with_capacity(input.value_dim);
        for (value_index, (&value, state_projection)) in v_row
            .iter()
            .zip(state_times_key.iter().copied())
            .enumerate()
        {
            let delta_value = beta * (value - state_projection);
            ensure_finite(delta_value, "delta", value_index)?;
            delta.push(delta_value);
        }

        for (key_index, key_value) in k_row.iter().copied().enumerate() {
            for (value_index, delta_value) in delta.iter().copied().enumerate() {
                let state_index = matrix_index(key_index, value_index, input.value_dim)?;
                let state_len = state.len();
                let state_value = state.get_mut(state_index).ok_or_else(|| {
                    LengthMismatchSnafu {
                        input: "state",
                        expected: state_index + 1,
                        actual: state_len,
                    }
                    .build()
                })?;
                *state_value += key_value * delta_value;
                ensure_finite(*state_value, "state update", state_index)?;
            }
        }

        for value_index in 0..input.value_dim {
            let mut accumulator = 0.0_f32;
            for (key_index, query_value) in q_row.iter().copied().enumerate() {
                let state_index = matrix_index(key_index, value_index, input.value_dim)?;
                let state_value = read_scalar(&state, state_index, "state", state.len())?;
                accumulator += state_value * query_value * input.scale;
                ensure_finite(accumulator, "output accumulation", value_index)?;
            }
            let output_index = checked_add(v_start, value_index, "output index")?;
            let output_len = output.len();
            let output_slot = output.get_mut(output_index).ok_or_else(|| {
                LengthMismatchSnafu {
                    input: "output",
                    expected: output_index + 1,
                    actual: output_len,
                }
                .build()
            })?;
            *output_slot = accumulator;
        }
    }

    Ok(RecurrentOutput { output, state })
}

fn checked_product(left: usize, right: usize, dimensions: &'static str) -> GdnResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| DimensionProductOverflowSnafu { dimensions }.build())
}

fn checked_add(left: usize, right: usize, dimensions: &'static str) -> GdnResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| DimensionProductOverflowSnafu { dimensions }.build())
}

fn matrix_index(key_index: usize, value_index: usize, value_dim: usize) -> GdnResult<usize> {
    checked_add(
        checked_product(key_index, value_dim, "key index * value_dim")?,
        value_index,
        "state matrix index",
    )
}

fn read_row<'a>(
    values: &'a [f32],
    start: usize,
    end: usize,
    input: &'static str,
) -> GdnResult<&'a [f32]> {
    values.get(start..end).ok_or_else(|| {
        LengthMismatchSnafu {
            input,
            expected: end,
            actual: values.len(),
        }
        .build()
    })
}

fn read_scalar(
    values: &[f32],
    index: usize,
    input: &'static str,
    expected: usize,
) -> GdnResult<f32> {
    values.get(index).copied().ok_or_else(|| {
        LengthMismatchSnafu {
            input,
            expected,
            actual: values.len(),
        }
        .build()
    })
}

fn validate_length(input: &'static str, actual: usize, expected: usize) -> GdnResult<()> {
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

fn validate_scalars(input: &'static str, values: &[f32]) -> GdnResult<()> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return NonFiniteInputSnafu { input, index }.fail();
        }
    }
    Ok(())
}

fn ensure_finite(value: f32, stage: &'static str, index: usize) -> GdnResult<()> {
    if !value.is_finite() {
        return NonFiniteArithmeticSnafu { stage, index }.fail();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_DIM: usize = 2;
    const VALUE_DIM: usize = 3;
    const SCALE: f32 = 0.5;
    const ORACLE_TOLERANCE: f32 = 1e-5;

    #[test]
    fn recurrence_matches_independent_f64_oracle() -> GdnResult<()> {
        let q = [0.25, -0.5, 1.25, 0.75, -1.0, 0.125];
        let k = [0.5, 1.0, -0.75, 0.25, 1.5, -0.5];
        let v = [1.0, -2.0, 0.5, -0.25, 1.25, 2.0, 0.75, -1.5, 0.125];
        let beta = [0.25, 0.75, 0.5];
        let g = [-0.125, 0.0625, -0.25];
        let state = [0.25, -0.5, 1.0, -1.25, 0.75, 0.125];

        let input = RecurrentInput::new(&q, &k, &v, &beta, &g, SCALE, &state, KEY_DIM, VALUE_DIM)?;
        let actual = recurrent_fwd(&input)?;
        let (expected_output, expected_state) =
            oracle_recurrence(&q, &k, &v, &beta, &g, SCALE, &state, KEY_DIM, VALUE_DIM);

        assert_close(actual.output(), &expected_output, "output");
        assert_close(actual.state(), &expected_state, "state");
        Ok(())
    }

    #[test]
    fn one_token_hand_case_updates_state_and_output() -> GdnResult<()> {
        let q = [1.0, 2.0];
        let k = [3.0, 4.0];
        let v = [5.0, -1.0];
        let beta = [1.0];
        let g = [0.0];
        let state = [0.0; 4];
        let input = RecurrentInput::new(&q, &k, &v, &beta, &g, 1.0, &state, 2, 2)?;
        let actual = recurrent_fwd(&input)?;

        assert_eq!(
            actual.state(),
            &[15.0, -3.0, 20.0, -4.0],
            "outer-product state mismatch"
        );
        assert_eq!(
            actual.output(),
            &[55.0, -11.0],
            "updated-state output mismatch"
        );
        Ok(())
    }

    #[test]
    fn recurrence_state_crosses_chunk_boundary() -> GdnResult<()> {
        let q = [0.25, -0.5, 1.25, 0.75, -1.0, 0.125];
        let k = [0.5, 1.0, -0.75, 0.25, 1.5, -0.5];
        let v = [1.0, -2.0, 0.5, -0.25, 1.25, 2.0, 0.75, -1.5, 0.125];
        let beta = [0.25, 0.75, 0.5];
        let g = [-0.125, 0.0625, -0.25];
        let state = [0.25, -0.5, 1.0, -1.25, 0.75, 0.125];
        let full_input =
            RecurrentInput::new(&q, &k, &v, &beta, &g, SCALE, &state, KEY_DIM, VALUE_DIM)?;
        let full = recurrent_fwd(&full_input)?;

        let first_input = RecurrentInput::new(
            &q[..KEY_DIM],
            &k[..KEY_DIM],
            &v[..VALUE_DIM],
            &beta[..1],
            &g[..1],
            SCALE,
            &state,
            KEY_DIM,
            VALUE_DIM,
        )?;
        let first = recurrent_fwd(&first_input)?;
        let rest_input = RecurrentInput::new(
            &q[KEY_DIM..],
            &k[KEY_DIM..],
            &v[VALUE_DIM..],
            &beta[1..],
            &g[1..],
            SCALE,
            first.state(),
            KEY_DIM,
            VALUE_DIM,
        )?;
        let rest = recurrent_fwd(&rest_input)?;

        let mut joined_output = first.output().to_vec();
        joined_output.extend_from_slice(rest.output());
        assert_eq!(
            full.output(),
            joined_output,
            "chunked output must equal full recurrence"
        );
        assert_eq!(
            full.state(),
            rest.state(),
            "chunked state must equal full recurrence"
        );
        Ok(())
    }

    #[test]
    fn malformed_and_nonfinite_inputs_are_rejected() {
        let finite = [1.0_f32; 4];
        let mismatched = RecurrentInput::new(
            &finite[..3],
            &finite,
            &finite,
            &[1.0],
            &[0.0],
            1.0,
            &finite,
            2,
            2,
        );
        assert!(
            matches!(mismatched, Err(GdnError::LengthMismatch { input: "q", .. })),
            "mismatched q length must fail admission"
        );

        let nonfinite = RecurrentInput::new(
            &finite[..2],
            &finite[..2],
            &finite[..2],
            &[f32::NAN],
            &[0.0],
            1.0,
            &finite,
            2,
            2,
        );
        assert!(
            matches!(
                nonfinite,
                Err(GdnError::NonFiniteInput { input: "beta", .. })
            ),
            "non-finite beta must fail admission"
        );

        let zero_dimension = RecurrentInput::new(&[], &[], &[], &[], &[], 1.0, &[], 0, 1);
        assert!(
            matches!(
                zero_dimension,
                Err(GdnError::ZeroDimension {
                    dimension: "key_dim",
                    ..
                })
            ),
            "zero key dimension must fail admission"
        );

        let overflowing_shape =
            RecurrentInput::new(&[], &[], &[], &[1.0, 1.0], &[0.0], 1.0, &[], usize::MAX, 2);
        assert!(
            matches!(
                overflowing_shape,
                Err(GdnError::DimensionProductOverflow {
                    dimensions: "token_count * key_dim",
                    ..
                })
            ),
            "overflowing dimension product must fail admission"
        );
    }

    #[test]
    fn arithmetic_overflow_returns_no_partial_result() -> GdnResult<()> {
        let input = RecurrentInput::new(
            &[1.0],
            &[1.0],
            &[1.0],
            &[1.0],
            &[1.0],
            1.0,
            &[f32::MAX],
            1,
            1,
        )?;
        let result = recurrent_fwd(&input);
        assert!(
            matches!(
                result,
                Err(GdnError::NonFiniteArithmetic {
                    stage: "state decay",
                    ..
                })
            ),
            "overflow must fail instead of returning a partial recurrence"
        );
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the independent oracle mirrors the fixed recurrence contract without sharing production validation helpers"
    )]
    fn oracle_recurrence(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        g: &[f32],
        scale: f32,
        state: &[f32],
        key_dim: usize,
        value_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut oracle_state: Vec<f64> = state.iter().copied().map(f64::from).collect();
        let mut oracle_output = Vec::with_capacity(beta.len() * value_dim);

        for token_index in 0..beta.len() {
            let decay = f64::from(g[token_index]).exp();
            for state_value in &mut oracle_state {
                *state_value *= decay;
            }

            let mut state_times_key = vec![0.0_f64; value_dim];
            for key_index in 0..key_dim {
                for value_index in 0..value_dim {
                    state_times_key[value_index] += oracle_state
                        [key_index * value_dim + value_index]
                        * f64::from(k[token_index * key_dim + key_index]);
                }
            }

            let mut delta = vec![0.0_f64; value_dim];
            for value_index in 0..value_dim {
                delta[value_index] = f64::from(beta[token_index])
                    * (f64::from(v[token_index * value_dim + value_index])
                        - state_times_key[value_index]);
            }

            for key_index in 0..key_dim {
                for value_index in 0..value_dim {
                    oracle_state[key_index * value_dim + value_index] +=
                        f64::from(k[token_index * key_dim + key_index]) * delta[value_index];
                }
            }

            for value_index in 0..value_dim {
                let mut accumulator = 0.0_f64;
                for key_index in 0..key_dim {
                    accumulator += oracle_state[key_index * value_dim + value_index]
                        * f64::from(q[token_index * key_dim + key_index])
                        * f64::from(scale);
                }
                oracle_output.push(accumulator as f32);
            }
        }

        (
            oracle_output,
            oracle_state.into_iter().map(|value| value as f32).collect(),
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual_value - expected_value).abs() <= ORACLE_TOLERANCE,
                "{label}[{index}] differs: actual={actual_value}, expected={expected_value}"
            );
        }
    }
}
