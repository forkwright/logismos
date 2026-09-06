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

    /// Value heads could not be assigned evenly to key/query heads.
    #[snafu(display(
        "{GDN_RECURRENCE}: value_head_count {value_head_count} is not divisible by key_head_count {key_head_count}"
    ))]
    HeadGroupingMismatch {
        /// Number of key/query heads.
        key_head_count: usize,
        /// Number of value heads.
        value_head_count: usize,
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

/// Validated head-major input for a grouped multi-head GDN recurrence.
///
/// The supported domain is `q/k: [Hk, T, K]`, `v: [Hv, T, V]`,
/// `beta/g: [Hv, T]`, and `state: [Hv, K, V]`, all `f32`. `Hv` must
/// divide evenly by `Hk`; value head `h` uses key/query head
/// `h / (Hv / Hk)`. `g` is natural-log decay and `scale` is passed unchanged
/// to each one-head recurrence.
///
/// This is an operator contract, not a model adapter. In particular, it does
/// not identify a model's tensor layout, normalization, projection, cache, or
/// precision policy.
#[derive(Debug, Clone, Copy)]
pub struct MultiHeadRecurrentInput<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    beta: &'a [f32],
    g: &'a [f32],
    scale: f32,
    state: &'a [f32],
    token_count: usize,
    key_head_count: usize,
    value_head_count: usize,
    key_dim: usize,
    value_dim: usize,
}

impl<'a> MultiHeadRecurrentInput<'a> {
    /// Admit an exact dense grouped multi-head recurrence input.
    ///
    /// # Errors
    ///
    /// Returns [`GdnError`] when a dimension is zero, value heads cannot map
    /// evenly to key/query heads, a shape product overflows, a buffer length
    /// differs from its declared shape, or any supplied scalar is non-finite.
    #[expect(
        clippy::too_many_arguments,
        reason = "the six input buffers, scale, token count, head counts and dimensions form the fixed grouped recurrence contract"
    )]
    pub fn new(
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        beta: &'a [f32],
        g: &'a [f32],
        scale: f32,
        state: &'a [f32],
        token_count: usize,
        key_head_count: usize,
        value_head_count: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> GdnResult<Self> {
        validate_nonzero_dimension("token_count", token_count)?;
        validate_nonzero_dimension("key_head_count", key_head_count)?;
        validate_nonzero_dimension("value_head_count", value_head_count)?;
        validate_nonzero_dimension("key_dim", key_dim)?;
        validate_nonzero_dimension("value_dim", value_dim)?;
        if !value_head_count.is_multiple_of(key_head_count) {
            return HeadGroupingMismatchSnafu {
                key_head_count,
                value_head_count,
            }
            .fail();
        }

        let key_head_width = checked_product(token_count, key_dim, "token_count * key_dim")?;
        let value_head_width = checked_product(token_count, value_dim, "token_count * value_dim")?;
        let state_head_width = checked_product(key_dim, value_dim, "key_dim * value_dim")?;
        validate_length(
            "q",
            q.len(),
            checked_product(
                key_head_count,
                key_head_width,
                "key_head_count * token_count * key_dim",
            )?,
        )?;
        validate_length(
            "k",
            k.len(),
            checked_product(
                key_head_count,
                key_head_width,
                "key_head_count * token_count * key_dim",
            )?,
        )?;
        validate_length(
            "v",
            v.len(),
            checked_product(
                value_head_count,
                value_head_width,
                "value_head_count * token_count * value_dim",
            )?,
        )?;
        validate_length(
            "beta",
            beta.len(),
            checked_product(
                value_head_count,
                token_count,
                "value_head_count * token_count",
            )?,
        )?;
        validate_length(
            "g",
            g.len(),
            checked_product(
                value_head_count,
                token_count,
                "value_head_count * token_count",
            )?,
        )?;
        validate_length(
            "state",
            state.len(),
            checked_product(
                value_head_count,
                state_head_width,
                "value_head_count * key_dim * value_dim",
            )?,
        )?;

        let input = Self {
            q,
            k,
            v,
            beta,
            g,
            scale,
            state,
            token_count,
            key_head_count,
            value_head_count,
            key_dim,
            value_dim,
        };
        input.validate_all_heads()?;
        Ok(input)
    }

    fn validate_all_heads(&self) -> GdnResult<()> {
        for value_head_index in 0..self.value_head_count {
            self.head_input(value_head_index)?;
        }
        Ok(())
    }

    fn head_input(&self, value_head_index: usize) -> GdnResult<RecurrentInput<'a>> {
        let heads_per_key = self.value_head_count / self.key_head_count;
        let key_head_index = value_head_index / heads_per_key;
        let key_head_width =
            checked_product(self.token_count, self.key_dim, "token_count * key_dim")?;
        let value_head_width =
            checked_product(self.token_count, self.value_dim, "token_count * value_dim")?;
        let state_head_width =
            checked_product(self.key_dim, self.value_dim, "key_dim * value_dim")?;
        RecurrentInput::new(
            head_slice(self.q, key_head_index, key_head_width, "q")?,
            head_slice(self.k, key_head_index, key_head_width, "k")?,
            head_slice(self.v, value_head_index, value_head_width, "v")?,
            head_slice(self.beta, value_head_index, self.token_count, "beta")?,
            head_slice(self.g, value_head_index, self.token_count, "g")?,
            self.scale,
            head_slice(self.state, value_head_index, state_head_width, "state")?,
            self.key_dim,
            self.value_dim,
        )
    }
}

/// Output and final state from [`multi_head_recurrent_fwd`].
#[derive(Debug, Clone, PartialEq)]
pub struct MultiHeadRecurrentOutput {
    output: Vec<f32>,
    state: Vec<f32>,
}

impl MultiHeadRecurrentOutput {
    /// Return the dense head-major `[Hv, T, V]` output.
    #[must_use]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Return the final head-major `[Hv, K, V]` state.
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

/// Evaluate the bounded grouped multi-head GDN reference.
///
/// Each value head is evaluated by the one-head recurrence using its grouped
/// key/query head. Inputs are immutable and every head is admitted before any
/// output execution begins; failures return no partial output or state.
///
/// # Errors
///
/// Returns [`GdnError`] when a checked one-head recurrence rejects an input or
/// produces non-finite arithmetic.
pub fn multi_head_recurrent_fwd(
    input: &MultiHeadRecurrentInput<'_>,
) -> GdnResult<MultiHeadRecurrentOutput> {
    input.validate_all_heads()?;
    let output_head_width = checked_product(
        input.token_count,
        input.value_dim,
        "token_count * value_dim",
    )?;
    let state_head_width = checked_product(input.key_dim, input.value_dim, "key_dim * value_dim")?;
    let output_capacity = checked_product(
        input.value_head_count,
        output_head_width,
        "value_head_count * token_count * value_dim",
    )?;
    let state_capacity = checked_product(
        input.value_head_count,
        state_head_width,
        "value_head_count * key_dim * value_dim",
    )?;
    let mut output = Vec::with_capacity(output_capacity);
    let mut state = Vec::with_capacity(state_capacity);

    for value_head_index in 0..input.value_head_count {
        let head_output = recurrent_fwd(&input.head_input(value_head_index)?)?;
        output.extend_from_slice(head_output.output());
        state.extend_from_slice(head_output.state());
    }

    Ok(MultiHeadRecurrentOutput { output, state })
}

fn checked_product(left: usize, right: usize, dimensions: &'static str) -> GdnResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| DimensionProductOverflowSnafu { dimensions }.build())
}

fn validate_nonzero_dimension(dimension: &'static str, value: usize) -> GdnResult<()> {
    if value == 0 {
        return ZeroDimensionSnafu { dimension }.fail();
    }
    Ok(())
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

fn head_slice<'a>(
    values: &'a [f32],
    head_index: usize,
    head_width: usize,
    input: &'static str,
) -> GdnResult<&'a [f32]> {
    let start = checked_product(head_index, head_width, "head index * head width")?;
    let end = checked_add(start, head_width, "head end")?;
    read_row(values, start, end, input)
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

    const KEY_HEAD_COUNT: usize = 2;
    const VALUE_HEAD_COUNT: usize = 4;
    const TOKEN_COUNT: usize = 3;
    const MULTI_HEAD_VALUE_DIM: usize = 2;

    struct MultiHeadFixture {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        beta: Vec<f32>,
        g: Vec<f32>,
        state: Vec<f32>,
    }

    #[test]
    fn multi_head_recurrence_matches_independent_f64_gqa_oracle() -> GdnResult<()> {
        let fixture = multi_head_fixture();
        let input = multi_head_input(
            &fixture.q,
            &fixture.k,
            &fixture.v,
            &fixture.beta,
            &fixture.g,
            &fixture.state,
            TOKEN_COUNT,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
        )?;
        let actual = multi_head_recurrent_fwd(&input)?;
        let (expected_output, expected_state) = oracle_multi_head_recurrence(
            &fixture.q,
            &fixture.k,
            &fixture.v,
            &fixture.beta,
            &fixture.g,
            SCALE,
            &fixture.state,
            TOKEN_COUNT,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
            KEY_DIM,
            MULTI_HEAD_VALUE_DIM,
        );

        assert_close(actual.output(), &expected_output, "multi-head output");
        assert_close(actual.state(), &expected_state, "multi-head state");
        Ok(())
    }

    #[test]
    fn multi_head_recurrence_state_crosses_chunk_boundary() -> GdnResult<()> {
        let fixture = multi_head_fixture();
        let full_input = multi_head_input(
            &fixture.q,
            &fixture.k,
            &fixture.v,
            &fixture.beta,
            &fixture.g,
            &fixture.state,
            TOKEN_COUNT,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
        )?;
        let full = multi_head_recurrent_fwd(&full_input)?;

        const FIRST_CHUNK_TOKENS: usize = 1;
        let first_q = head_major_token_window(
            &fixture.q,
            KEY_HEAD_COUNT,
            TOKEN_COUNT,
            KEY_DIM,
            0,
            FIRST_CHUNK_TOKENS,
        );
        let first_k = head_major_token_window(
            &fixture.k,
            KEY_HEAD_COUNT,
            TOKEN_COUNT,
            KEY_DIM,
            0,
            FIRST_CHUNK_TOKENS,
        );
        let first_v = head_major_token_window(
            &fixture.v,
            VALUE_HEAD_COUNT,
            TOKEN_COUNT,
            MULTI_HEAD_VALUE_DIM,
            0,
            FIRST_CHUNK_TOKENS,
        );
        let first_beta = head_major_token_window(
            &fixture.beta,
            VALUE_HEAD_COUNT,
            TOKEN_COUNT,
            1,
            0,
            FIRST_CHUNK_TOKENS,
        );
        let first_g = head_major_token_window(
            &fixture.g,
            VALUE_HEAD_COUNT,
            TOKEN_COUNT,
            1,
            0,
            FIRST_CHUNK_TOKENS,
        );
        let first_input = multi_head_input(
            &first_q,
            &first_k,
            &first_v,
            &first_beta,
            &first_g,
            &fixture.state,
            FIRST_CHUNK_TOKENS,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
        )?;
        let first = multi_head_recurrent_fwd(&first_input)?;

        let remaining_token_count = TOKEN_COUNT - FIRST_CHUNK_TOKENS;
        let rest_q = head_major_token_window(
            &fixture.q,
            KEY_HEAD_COUNT,
            TOKEN_COUNT,
            KEY_DIM,
            FIRST_CHUNK_TOKENS,
            TOKEN_COUNT,
        );
        let rest_k = head_major_token_window(
            &fixture.k,
            KEY_HEAD_COUNT,
            TOKEN_COUNT,
            KEY_DIM,
            FIRST_CHUNK_TOKENS,
            TOKEN_COUNT,
        );
        let rest_v = head_major_token_window(
            &fixture.v,
            VALUE_HEAD_COUNT,
            TOKEN_COUNT,
            MULTI_HEAD_VALUE_DIM,
            FIRST_CHUNK_TOKENS,
            TOKEN_COUNT,
        );
        let rest_beta = head_major_token_window(
            &fixture.beta,
            VALUE_HEAD_COUNT,
            TOKEN_COUNT,
            1,
            FIRST_CHUNK_TOKENS,
            TOKEN_COUNT,
        );
        let rest_g = head_major_token_window(
            &fixture.g,
            VALUE_HEAD_COUNT,
            TOKEN_COUNT,
            1,
            FIRST_CHUNK_TOKENS,
            TOKEN_COUNT,
        );
        let rest_input = multi_head_input(
            &rest_q,
            &rest_k,
            &rest_v,
            &rest_beta,
            &rest_g,
            first.state(),
            remaining_token_count,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
        )?;
        let rest = multi_head_recurrent_fwd(&rest_input)?;
        let joined_output = join_head_major_outputs(
            first.output(),
            rest.output(),
            VALUE_HEAD_COUNT,
            FIRST_CHUNK_TOKENS,
            remaining_token_count,
            MULTI_HEAD_VALUE_DIM,
        );

        assert_eq!(
            full.output(),
            joined_output,
            "chunked multi-head output must equal full recurrence"
        );
        assert_eq!(
            full.state(),
            rest.state(),
            "chunked multi-head state must equal full recurrence"
        );
        Ok(())
    }

    #[test]
    fn multi_head_recurrence_supports_equal_key_and_value_head_counts() -> GdnResult<()> {
        const EQUAL_HEAD_COUNT: usize = 2;
        const EQUAL_TOKEN_COUNT: usize = 2;
        const EQUAL_VALUE_DIM: usize = 1;
        let q = fixture_values(EQUAL_HEAD_COUNT * EQUAL_TOKEN_COUNT * KEY_DIM, -0.25);
        let k = fixture_values(EQUAL_HEAD_COUNT * EQUAL_TOKEN_COUNT * KEY_DIM, 0.5);
        let v = fixture_values(
            EQUAL_HEAD_COUNT * EQUAL_TOKEN_COUNT * EQUAL_VALUE_DIM,
            -0.75,
        );
        let beta = beta_values(EQUAL_HEAD_COUNT * EQUAL_TOKEN_COUNT);
        let g = gate_values(EQUAL_HEAD_COUNT * EQUAL_TOKEN_COUNT);
        let state = fixture_values(EQUAL_HEAD_COUNT * KEY_DIM * EQUAL_VALUE_DIM, 0.125);
        let input = MultiHeadRecurrentInput::new(
            &q,
            &k,
            &v,
            &beta,
            &g,
            SCALE,
            &state,
            EQUAL_TOKEN_COUNT,
            EQUAL_HEAD_COUNT,
            EQUAL_HEAD_COUNT,
            KEY_DIM,
            EQUAL_VALUE_DIM,
        )?;
        let actual = multi_head_recurrent_fwd(&input)?;
        let (expected_output, expected_state) = oracle_multi_head_recurrence(
            &q,
            &k,
            &v,
            &beta,
            &g,
            SCALE,
            &state,
            EQUAL_TOKEN_COUNT,
            EQUAL_HEAD_COUNT,
            EQUAL_HEAD_COUNT,
            KEY_DIM,
            EQUAL_VALUE_DIM,
        );

        assert_close(actual.output(), &expected_output, "equal-head output");
        assert_close(actual.state(), &expected_state, "equal-head state");
        Ok(())
    }

    #[test]
    fn multi_head_admission_rejects_invalid_grouping_and_inputs() -> GdnResult<()> {
        let fixture = multi_head_fixture();
        let grouping = MultiHeadRecurrentInput::new(
            &fixture.q,
            &fixture.k,
            &fixture.v,
            &fixture.beta,
            &fixture.g,
            SCALE,
            &fixture.state,
            TOKEN_COUNT,
            KEY_HEAD_COUNT,
            3,
            KEY_DIM,
            MULTI_HEAD_VALUE_DIM,
        );
        assert!(
            matches!(grouping, Err(GdnError::HeadGroupingMismatch { .. })),
            "non-divisible value heads must fail admission"
        );

        let short_q = MultiHeadRecurrentInput::new(
            &fixture.q[..fixture.q.len() - 1],
            &fixture.k,
            &fixture.v,
            &fixture.beta,
            &fixture.g,
            SCALE,
            &fixture.state,
            TOKEN_COUNT,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
            KEY_DIM,
            MULTI_HEAD_VALUE_DIM,
        );
        assert!(
            matches!(short_q, Err(GdnError::LengthMismatch { input: "q", .. })),
            "short head-major q input must fail admission"
        );

        let zero_heads = MultiHeadRecurrentInput::new(
            &[],
            &[],
            &[],
            &[],
            &[],
            SCALE,
            &[],
            TOKEN_COUNT,
            0,
            VALUE_HEAD_COUNT,
            KEY_DIM,
            MULTI_HEAD_VALUE_DIM,
        );
        assert!(
            matches!(
                zero_heads,
                Err(GdnError::ZeroDimension {
                    dimension: "key_head_count",
                    ..
                })
            ),
            "zero key head count must fail admission"
        );

        let overflowing_shape = MultiHeadRecurrentInput::new(
            &[],
            &[],
            &[],
            &[],
            &[],
            SCALE,
            &[],
            usize::MAX,
            1,
            1,
            2,
            1,
        );
        assert!(
            matches!(
                overflowing_shape,
                Err(GdnError::DimensionProductOverflow {
                    dimensions: "token_count * key_dim",
                    ..
                })
            ),
            "overflowing multi-head shape must fail before allocation"
        );

        let mut nonfinite_g = fixture.g.clone();
        let last_gate = nonfinite_g.last_mut().ok_or_else(|| {
            LengthMismatchSnafu {
                input: "g",
                expected: 1_usize,
                actual: 0_usize,
            }
            .build()
        })?;
        *last_gate = f32::NAN;
        let nonfinite = MultiHeadRecurrentInput::new(
            &fixture.q,
            &fixture.k,
            &fixture.v,
            &fixture.beta,
            &nonfinite_g,
            SCALE,
            &fixture.state,
            TOKEN_COUNT,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
            KEY_DIM,
            MULTI_HEAD_VALUE_DIM,
        );
        assert!(
            matches!(nonfinite, Err(GdnError::NonFiniteInput { input: "g", .. })),
            "non-finite scalar in a later value head must fail admission"
        );
        Ok(())
    }

    #[test]
    fn multi_head_arithmetic_failure_preserves_caller_state() -> GdnResult<()> {
        let state = [f32::MAX, f32::MAX];
        let state_before = state;
        let input = MultiHeadRecurrentInput::new(
            &[1.0],
            &[1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            1.0,
            &state,
            1,
            1,
            2,
            1,
            1,
        )?;
        let result = multi_head_recurrent_fwd(&input);
        assert!(
            matches!(
                result,
                Err(GdnError::NonFiniteArithmetic {
                    stage: "state decay",
                    ..
                })
            ),
            "non-finite arithmetic must fail without returning a partial multi-head result"
        );
        assert!(
            state
                .iter()
                .zip(state_before)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "multi-head recurrence must not mutate caller-owned state"
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

    fn multi_head_fixture() -> MultiHeadFixture {
        MultiHeadFixture {
            q: fixture_values(KEY_HEAD_COUNT * TOKEN_COUNT * KEY_DIM, -0.5),
            k: fixture_values(KEY_HEAD_COUNT * TOKEN_COUNT * KEY_DIM, 0.25),
            v: fixture_values(VALUE_HEAD_COUNT * TOKEN_COUNT * MULTI_HEAD_VALUE_DIM, -0.75),
            beta: beta_values(VALUE_HEAD_COUNT * TOKEN_COUNT),
            g: gate_values(VALUE_HEAD_COUNT * TOKEN_COUNT),
            state: fixture_values(VALUE_HEAD_COUNT * KEY_DIM * MULTI_HEAD_VALUE_DIM, 0.125),
        }
    }

    fn fixture_values(element_count: usize, offset: f32) -> Vec<f32> {
        (0..element_count)
            .map(|index| (index as f32 + offset) * 0.125 - 0.5)
            .collect()
    }

    fn beta_values(element_count: usize) -> Vec<f32> {
        (0..element_count)
            .map(|index| 0.1 + index as f32 * 0.03)
            .collect()
    }

    fn gate_values(element_count: usize) -> Vec<f32> {
        (0..element_count)
            .map(|index| -0.4 + index as f32 * 0.025)
            .collect()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the helper fixes the common grouped fixture dimensions while accepting each independently owned head-major buffer"
    )]
    fn multi_head_input<'a>(
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        beta: &'a [f32],
        g: &'a [f32],
        state: &'a [f32],
        token_count: usize,
        key_head_count: usize,
        value_head_count: usize,
    ) -> GdnResult<MultiHeadRecurrentInput<'a>> {
        MultiHeadRecurrentInput::new(
            q,
            k,
            v,
            beta,
            g,
            SCALE,
            state,
            token_count,
            key_head_count,
            value_head_count,
            KEY_DIM,
            MULTI_HEAD_VALUE_DIM,
        )
    }

    fn head_major_token_window(
        values: &[f32],
        head_count: usize,
        token_count: usize,
        element_width: usize,
        start_token: usize,
        end_token: usize,
    ) -> Vec<f32> {
        let head_width = token_count * element_width;
        assert_eq!(
            values.len(),
            head_count * head_width,
            "fixture must have one contiguous sequence per head"
        );
        let mut window = Vec::with_capacity(head_count * (end_token - start_token) * element_width);
        for head_values in values.chunks_exact(head_width) {
            for token_values in head_values
                .chunks_exact(element_width)
                .skip(start_token)
                .take(end_token - start_token)
            {
                window.extend_from_slice(token_values);
            }
        }
        window
    }

    fn join_head_major_outputs(
        first: &[f32],
        rest: &[f32],
        value_head_count: usize,
        first_token_count: usize,
        rest_token_count: usize,
        value_dim: usize,
    ) -> Vec<f32> {
        let first_head_width = first_token_count * value_dim;
        let rest_head_width = rest_token_count * value_dim;
        assert_eq!(
            first.len(),
            value_head_count * first_head_width,
            "first output must be head-major"
        );
        assert_eq!(
            rest.len(),
            value_head_count * rest_head_width,
            "rest output must be head-major"
        );
        let mut joined = Vec::with_capacity(first.len() + rest.len());
        for (first_head, rest_head) in first
            .chunks_exact(first_head_width)
            .zip(rest.chunks_exact(rest_head_width))
        {
            joined.extend_from_slice(first_head);
            joined.extend_from_slice(rest_head);
        }
        joined
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the independent oracle spells out the complete grouped recurrence contract without reusing production validation or recurrence helpers"
    )]
    fn oracle_multi_head_recurrence(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        g: &[f32],
        scale: f32,
        state: &[f32],
        token_count: usize,
        key_head_count: usize,
        value_head_count: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let key_head_width = token_count * key_dim;
        let value_head_width = token_count * value_dim;
        let state_head_width = key_dim * value_dim;
        let heads_per_key = value_head_count / key_head_count;
        let mut output = Vec::with_capacity(value_head_count * value_head_width);
        let mut final_state = Vec::with_capacity(value_head_count * state_head_width);

        for value_head_index in 0..value_head_count {
            let key_head_index = value_head_index / heads_per_key;
            let q_head = &q[key_head_index * key_head_width..(key_head_index + 1) * key_head_width];
            let k_head = &k[key_head_index * key_head_width..(key_head_index + 1) * key_head_width];
            let v_head =
                &v[value_head_index * value_head_width..(value_head_index + 1) * value_head_width];
            let beta_head =
                &beta[value_head_index * token_count..(value_head_index + 1) * token_count];
            let g_head = &g[value_head_index * token_count..(value_head_index + 1) * token_count];
            let mut head_state: Vec<f64> = state
                [value_head_index * state_head_width..(value_head_index + 1) * state_head_width]
                .iter()
                .copied()
                .map(f64::from)
                .collect();

            for token_index in 0..token_count {
                let decay = f64::from(g_head[token_index]).exp();
                for state_value in &mut head_state {
                    *state_value *= decay;
                }

                let mut state_times_key = vec![0.0_f64; value_dim];
                for key_index in 0..key_dim {
                    for value_index in 0..value_dim {
                        state_times_key[value_index] += head_state
                            [key_index * value_dim + value_index]
                            * f64::from(k_head[token_index * key_dim + key_index]);
                    }
                }

                let mut delta = vec![0.0_f64; value_dim];
                for value_index in 0..value_dim {
                    delta[value_index] = f64::from(beta_head[token_index])
                        * (f64::from(v_head[token_index * value_dim + value_index])
                            - state_times_key[value_index]);
                }

                for key_index in 0..key_dim {
                    for value_index in 0..value_dim {
                        head_state[key_index * value_dim + value_index] +=
                            f64::from(k_head[token_index * key_dim + key_index])
                                * delta[value_index];
                    }
                }

                for value_index in 0..value_dim {
                    let mut accumulator = 0.0_f64;
                    for key_index in 0..key_dim {
                        accumulator += head_state[key_index * value_dim + value_index]
                            * f64::from(q_head[token_index * key_dim + key_index])
                            * f64::from(scale);
                    }
                    output.push(accumulator as f32);
                }
            }
            final_state.extend(head_state.into_iter().map(|value| value as f32));
        }

        (output, final_state)
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
