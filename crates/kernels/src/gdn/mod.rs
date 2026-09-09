//! Bounded CPU references for single-head and grouped Gated Delta Rule recurrence.
//!
//! This module accepts dense single-head and grouped multi-head recurrent input.
//! It provides a correctness oracle plus a staged device step, not a model
//! adapter or a permissive fallback for unsupported GDN variants.
//! Bounds describe admitted shapes and exact logical `f32` requests. Every
//! owned output or scratch vector is reserved fallibly; this is not a process
//! RSS, allocator-overhead, or physical-memory guarantee.

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
use hipcore::Stream;
use snafu::{ResultExt, Snafu};

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use crate::error::LaunchSnafu;
#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
use crate::error::NoGpuBuildSnafu;
#[cfg(feature = "gpu")]
use crate::error::Result;
#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
use crate::error::UnsupportedShapeSnafu;

const GDN_RECURRENCE: &str = "gdn_recurrent_fwd";
#[cfg(feature = "gpu")]
const GDN_GROUPED_STEP_KERNEL: &str = "gdn_grouped_step_f32";
#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
const MAX_GDN_VALUE_DIM: usize = 1024;

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe extern "C" {
    fn logismos_launch_gdn_grouped_step_f32(
        q_f32: *const c_void,
        k_f32: *const c_void,
        v_f32: *const c_void,
        beta_f32: *const c_void,
        g_f32: *const c_void,
        state_in_f32: *const c_void,
        state_out_f32: *mut c_void,
        output_f32: *mut c_void,
        scale: f32,
        key_head_count: u32,
        value_head_count: u32,
        key_dim: u32,
        value_dim: u32,
        stream: *mut c_void,
    ) -> u32;
}

/// Result alias for the bounded GDN reference.
pub type GdnResult<T> = core::result::Result<T, GdnError>;

/// Exact logical `f32` capacities owned by one grouped GDN evaluation.
///
/// The aggregate output and state coexist with one value head's output, state,
/// state-times-key projection, and delta. These are requested vector
/// capacities, not allocator capacity or resident memory.
///
/// WHY: callers can compose the kernel's allocation envelope without copying
/// its dimension arithmetic or inventing model-level coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiHeadRecurrentAllocationPlan {
    token_count: usize,
    key_head_count: usize,
    value_head_count: usize,
    key_dim: usize,
    value_dim: usize,
    query_and_key_elements: usize,
    output_elements: usize,
    scalar_elements: usize,
    state_elements: usize,
    head: RecurrentAllocationPlan,
    workspace_elements: usize,
}

impl MultiHeadRecurrentAllocationPlan {
    /// Derive the allocation requests for one grouped recurrence shape.
    ///
    /// # Errors
    ///
    /// Returns [`GdnError`] when a required dimension is zero, value heads do
    /// not divide evenly across key heads, or an element product or
    /// allocation-envelope sum overflows.
    pub fn try_from_dimensions(
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
        let head = RecurrentAllocationPlan::try_from_dimensions(token_count, key_dim, value_dim)?;
        let query_and_key_elements = checked_product(
            key_head_count,
            head.query_and_key,
            "key_head_count * token_count * key_dim",
        )?;
        let output_elements = checked_product(
            value_head_count,
            head.output,
            "value_head_count * token_count * value_dim",
        )?;
        let scalar_elements = checked_product(
            value_head_count,
            token_count,
            "value_head_count * token_count",
        )?;
        let state_elements = checked_product(
            value_head_count,
            head.state,
            "value_head_count * key_dim * value_dim",
        )?;
        let workspace_elements = [
            output_elements,
            state_elements,
            head.output,
            head.state,
            head.state_times_key,
            head.delta,
        ]
        .into_iter()
        .try_fold(0_usize, checked_allocation_sum)?;
        Ok(Self {
            token_count,
            key_head_count,
            value_head_count,
            key_dim,
            value_dim,
            query_and_key_elements,
            output_elements,
            scalar_elements,
            state_elements,
            head,
            workspace_elements,
        })
    }

    /// Return the aggregate head-major output request.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output_elements
    }

    /// Return the admitted token count.
    #[must_use]
    pub const fn token_count(self) -> usize {
        self.token_count
    }

    /// Return the admitted key/query-head count.
    #[must_use]
    pub const fn key_head_count(self) -> usize {
        self.key_head_count
    }

    /// Return the admitted value-head count.
    #[must_use]
    pub const fn value_head_count(self) -> usize {
        self.value_head_count
    }

    /// Return the admitted per-head key/query width.
    #[must_use]
    pub const fn key_dim(self) -> usize {
        self.key_dim
    }

    /// Return the admitted per-head value width.
    #[must_use]
    pub const fn value_dim(self) -> usize {
        self.value_dim
    }

    /// Return the aggregate final-state request.
    #[must_use]
    pub const fn state_elements(self) -> usize {
        self.state_elements
    }

    /// Return the aggregate head-major query/key request.
    #[must_use]
    pub const fn query_and_key_elements(self) -> usize {
        self.query_and_key_elements
    }

    /// Return the aggregate head-major beta or gate request.
    #[must_use]
    pub const fn scalar_elements(self) -> usize {
        self.scalar_elements
    }

    /// Return one concurrently evaluated head's output request.
    #[must_use]
    pub const fn head_output_elements(self) -> usize {
        self.head.output
    }

    /// Return one concurrently evaluated head's state request.
    #[must_use]
    pub const fn head_state_elements(self) -> usize {
        self.head.state
    }

    /// Return one token's state-times-key projection request.
    #[must_use]
    pub const fn state_times_key_elements(self) -> usize {
        self.head.state_times_key
    }

    /// Return one token's delta request.
    #[must_use]
    pub const fn delta_elements(self) -> usize {
        self.head.delta
    }

    /// Return the conservative sum of simultaneously owned GDN requests.
    #[must_use]
    pub const fn workspace_elements(self) -> usize {
        self.workspace_elements
    }

    const fn head_allocations(self) -> RecurrentAllocationPlan {
        self.head
    }
}

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

    /// Summing checked recurrence allocation requests exceeded `usize`.
    #[snafu(display("{GDN_RECURRENCE}: logical allocation sum overflows usize"))]
    DimensionSumOverflow {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A checked recurrence buffer could not reserve its exact capacity.
    #[snafu(display("{GDN_RECURRENCE}: allocation for {allocation} ({elements} elements) failed"))]
    Allocation {
        /// Named recurrence buffer.
        allocation: &'static str,
        /// Exact requested scalar count.
        elements: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
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
    allocations: RecurrentAllocationPlan,
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
        let token_count = beta.len();
        let allocations =
            RecurrentAllocationPlan::try_from_dimensions(token_count, key_dim, value_dim)?;
        Self::new_with_allocations(
            q,
            k,
            v,
            beta,
            g,
            scale,
            state,
            key_dim,
            value_dim,
            allocations,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the internal admission path additionally receives its owner-derived allocation plan"
    )]
    fn new_with_allocations(
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        beta: &'a [f32],
        g: &'a [f32],
        scale: f32,
        state: &'a [f32],
        key_dim: usize,
        value_dim: usize,
        allocations: RecurrentAllocationPlan,
    ) -> GdnResult<Self> {
        let token_count = beta.len();

        validate_length("q", q.len(), allocations.query_and_key)?;
        validate_length("k", k.len(), allocations.query_and_key)?;
        validate_length("v", v.len(), allocations.output)?;
        validate_length("g", g.len(), token_count)?;
        validate_length("state", state.len(), allocations.state)?;
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
            allocations,
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
    allocations: MultiHeadRecurrentAllocationPlan,
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
        let allocations = MultiHeadRecurrentAllocationPlan::try_from_dimensions(
            token_count,
            key_head_count,
            value_head_count,
            key_dim,
            value_dim,
        )?;

        validate_length("q", q.len(), allocations.query_and_key_elements())?;
        validate_length("k", k.len(), allocations.query_and_key_elements())?;
        validate_length("v", v.len(), allocations.output_elements())?;
        validate_length("beta", beta.len(), allocations.scalar_elements())?;
        validate_length("g", g.len(), allocations.scalar_elements())?;
        validate_length("state", state.len(), allocations.state_elements())?;

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
            allocations,
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
        let head = self.allocations.head_allocations();
        RecurrentInput::new_with_allocations(
            head_slice(self.q, key_head_index, head.query_and_key, "q")?,
            head_slice(self.k, key_head_index, head.query_and_key, "k")?,
            head_slice(self.v, value_head_index, head.output, "v")?,
            head_slice(self.beta, value_head_index, self.token_count, "beta")?,
            head_slice(self.g, value_head_index, self.token_count, "g")?,
            self.scale,
            head_slice(self.state, value_head_index, head.state, "state")?,
            self.key_dim,
            self.value_dim,
            head,
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
    let mut state = reserve_f32("one-head state", input.allocations.state)?;
    state.extend_from_slice(input.state);
    let mut output = reserve_f32("one-head output", input.allocations.output)?;
    output.resize(input.allocations.output, 0.0);

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

        let mut state_times_key =
            reserve_f32("state times key", input.allocations.state_times_key)?;
        state_times_key.resize(input.allocations.state_times_key, 0.0);
        for (key_index, key_value) in k_row.iter().copied().enumerate() {
            for (value_index, accumulator) in state_times_key.iter_mut().enumerate() {
                let state_index = matrix_index(key_index, value_index, input.value_dim)?;
                let state_value = read_scalar(&state, state_index, "state", state.len())?;
                *accumulator += state_value * key_value;
                ensure_finite(*accumulator, "state times key", value_index)?;
            }
        }

        let mut delta = reserve_f32("delta", input.allocations.delta)?;
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
    let mut output = reserve_f32("multi-head output", input.allocations.output_elements())?;
    let mut state = reserve_f32("multi-head state", input.allocations.state_elements())?;

    for value_head_index in 0..input.value_head_count {
        let head_output = recurrent_fwd(&input.head_input(value_head_index)?)?;
        output.extend_from_slice(head_output.output());
        state.extend_from_slice(head_output.state());
    }

    Ok(MultiHeadRecurrentOutput { output, state })
}

#[cfg(feature = "gpu")]
/// Launch one staged, grouped, dense-f32 GDN decode step on `stream`.
///
/// `plan` owns the sole shape interpretation: the admitted step has `T = 1`,
/// `q/k: [Hk, K]`, `v: [Hv, V]`, `beta/g: [Hv]`, `state_in/out: [Hv, K, V]`,
/// and `output: [Hv, V]`. Value head `h` reads key/query head
/// `h / (Hv / Hk)`. `state_in_f32` remains immutable; the kernel writes all
/// of `state_out_f32` and `output_f32` as separate staged results. It is not a
/// decoder, cache, or active-session integration path.
///
/// The one-thread-per-value-column implementation serializes the key axis.
/// It uses natural `exp(g)` and, like [`recurrent_fwd`], adds
/// `(state * q) * scale` for each key term rather than scaling a completed
/// sum. The source-scoped HIP flags disable fast math and contraction.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] when `plan` is not a one-token
/// dense-f32 step, the value width exceeds this baseline's one-block profile,
/// a declared buffer length differs from `plan`, a span is null, unaligned,
/// unrepresentable, overlaps a writable result, or a dimension cannot cross
/// the `u32` HIP ABI. A CPU-only build returns [`crate::Error::NoGpuBuild`]
/// without initializing HIP. It propagates stream-current failures and
/// returns [`crate::Error::Launch`] when HIP rejects the kernel submission.
///
/// # Safety
///
/// Each pointer must designate a live allocation on `stream`'s device for the
/// exact declared `f32` element count through stream completion. The two
/// writable spans must not alias each other or any input; inputs may alias
/// other inputs. Both writable spans require exclusive access through stream
/// completion: no other GPU command or host alias may read or write either
/// span. No producer may modify any input through stream completion.
///
/// Device contents are not inspectable at this boundary. Callers must ensure
/// every input, `scale`, and every recurrence intermediate is finite and
/// either zero or normal `f32`; in particular, `exp(g)`, next-state values,
/// and output values must remain finite. Subnormal-dependent behavior is not
/// qualified. The kernel has no status channel and therefore cannot reproduce
/// the CPU reference's non-finite-input or arithmetic refusals.
#[expect(
    clippy::too_many_arguments,
    reason = "the eight buffers, scale, and stream are the fixed staged GDN step ABI"
)]
pub unsafe fn launch_multi_head_recurrent_step_f32(
    plan: MultiHeadRecurrentAllocationPlan,
    q_f32: *const f32,
    q_elements: usize,
    k_f32: *const f32,
    k_elements: usize,
    v_f32: *const f32,
    v_elements: usize,
    beta_f32: *const f32,
    beta_elements: usize,
    g_f32: *const f32,
    g_elements: usize,
    scale: f32,
    state_in_f32: *const f32,
    state_in_elements: usize,
    state_out_f32: *mut f32,
    state_out_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            q_f32,
            q_elements,
            k_f32,
            k_elements,
            v_f32,
            v_elements,
            beta_f32,
            beta_elements,
            g_f32,
            g_elements,
            scale,
            state_in_f32,
            state_in_elements,
            state_out_f32,
            state_out_elements,
            output_f32,
            output_elements,
            stream,
        );
        no_gpu_gdn_step_refusal()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let abi = validate_gdn_step_launch(
            plan,
            q_f32,
            q_elements,
            k_f32,
            k_elements,
            v_f32,
            v_elements,
            beta_f32,
            beta_elements,
            g_f32,
            g_elements,
            scale,
            state_in_f32,
            state_in_elements,
            state_out_f32,
            state_out_elements,
            output_f32,
            output_elements,
        )?;
        stream.make_current()?;
        // SAFETY: the caller upholds device ownership, lifetime, concurrent
        // access, and numerical-domain obligations documented above; checked
        // spans and the allocation-plan owner established exact extents,
        // alignment, non-aliasing results, and ABI dimensions.
        let code = unsafe {
            logismos_launch_gdn_grouped_step_f32(
                q_f32.cast::<c_void>(),
                k_f32.cast::<c_void>(),
                v_f32.cast::<c_void>(),
                beta_f32.cast::<c_void>(),
                g_f32.cast::<c_void>(),
                state_in_f32.cast::<c_void>(),
                state_out_f32.cast::<c_void>(),
                output_f32.cast::<c_void>(),
                scale,
                abi.key_head_count,
                abi.value_head_count,
                abi.key_dim,
                abi.value_dim,
                stream.raw().cast::<c_void>(),
            )
        };
        if code == 0 {
            Ok(())
        } else {
            LaunchSnafu {
                kernel: GDN_GROUPED_STEP_KERNEL,
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            }
            .fail()
        }
    }
}

#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
fn no_gpu_gdn_step_refusal() -> Result<()> {
    NoGpuBuildSnafu {
        kernel: GDN_GROUPED_STEP_KERNEL,
    }
    .fail()
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
#[derive(Clone, Copy)]
struct GdnStepAbi {
    key_head_count: u32,
    value_head_count: u32,
    key_dim: u32,
    value_dim: u32,
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
#[derive(Clone, Copy)]
struct DeviceSpan {
    start: usize,
    end: usize,
    name: &'static str,
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
#[expect(
    clippy::too_many_arguments,
    reason = "validation receives the fixed raw staged GDN step ABI without constructing a second shape owner"
)]
fn validate_gdn_step_launch(
    plan: MultiHeadRecurrentAllocationPlan,
    q_f32: *const f32,
    q_elements: usize,
    k_f32: *const f32,
    k_elements: usize,
    v_f32: *const f32,
    v_elements: usize,
    beta_f32: *const f32,
    beta_elements: usize,
    g_f32: *const f32,
    g_elements: usize,
    scale: f32,
    state_in_f32: *const f32,
    state_in_elements: usize,
    state_out_f32: *mut f32,
    state_out_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
) -> Result<GdnStepAbi> {
    if plan.token_count() != 1 {
        return unsupported_gdn_step_shape(format!(
            "only dense-f32 T=1 decode steps are supported, got T={}",
            plan.token_count()
        ));
    }
    if plan.value_dim() > MAX_GDN_VALUE_DIM {
        return unsupported_gdn_step_shape(format!(
            "value_dim {} exceeds the one-block dense-f32 profile limit {MAX_GDN_VALUE_DIM}",
            plan.value_dim()
        ));
    }
    if !(scale == 0.0 || scale.is_normal()) {
        return unsupported_gdn_step_shape(
            "scale must be zero or normal finite f32 for the declared device-input domain"
                .to_string(),
        );
    }

    validate_gdn_step_length("q", q_elements, plan.query_and_key_elements())?;
    validate_gdn_step_length("k", k_elements, plan.query_and_key_elements())?;
    validate_gdn_step_length("v", v_elements, plan.output_elements())?;
    validate_gdn_step_length("beta", beta_elements, plan.scalar_elements())?;
    validate_gdn_step_length("g", g_elements, plan.scalar_elements())?;
    validate_gdn_step_length("state_in", state_in_elements, plan.state_elements())?;
    validate_gdn_step_length("state_out", state_out_elements, plan.state_elements())?;
    validate_gdn_step_length("output", output_elements, plan.output_elements())?;

    let inputs = [
        checked_device_span(q_f32, q_elements, "q")?,
        checked_device_span(k_f32, k_elements, "k")?,
        checked_device_span(v_f32, v_elements, "v")?,
        checked_device_span(beta_f32, beta_elements, "beta")?,
        checked_device_span(g_f32, g_elements, "g")?,
        checked_device_span(state_in_f32, state_in_elements, "state_in")?,
    ];
    let state_out =
        checked_device_span(state_out_f32.cast_const(), state_out_elements, "state_out")?;
    let output = checked_device_span(output_f32.cast_const(), output_elements, "output")?;
    for input in inputs {
        reject_overlapping_gdn_step_spans(state_out, input)?;
        reject_overlapping_gdn_step_spans(output, input)?;
    }
    reject_overlapping_gdn_step_spans(state_out, output)?;

    Ok(GdnStepAbi {
        key_head_count: gdn_step_u32("key_head_count", plan.key_head_count())?,
        value_head_count: gdn_step_u32("value_head_count", plan.value_head_count())?,
        key_dim: gdn_step_u32("key_dim", plan.key_dim())?,
        value_dim: gdn_step_u32("value_dim", plan.value_dim())?,
    })
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn validate_gdn_step_length(name: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        unsupported_gdn_step_shape(format!(
            "{name} length {actual} does not match allocation-plan extent {expected}"
        ))
    }
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn checked_device_span(
    pointer: *const f32,
    elements: usize,
    name: &'static str,
) -> Result<DeviceSpan> {
    if pointer.is_null() {
        return unsupported_gdn_step_shape(format!("{name} must be non-null"));
    }
    if !pointer.addr().is_multiple_of(core::mem::align_of::<f32>()) {
        return unsupported_gdn_step_shape(format!("{name} must be aligned for f32"));
    }
    let layout = std::alloc::Layout::array::<f32>(elements).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: GDN_GROUPED_STEP_KERNEL,
            msg: format!("{name} length {elements} exceeds the Rust allocation layout domain"),
        }
        .build()
    })?;
    let start = pointer.addr();
    let end = start.checked_add(layout.size()).ok_or_else(|| {
        UnsupportedShapeSnafu {
            kernel: GDN_GROUPED_STEP_KERNEL,
            msg: format!("{name} device span overflows the address domain"),
        }
        .build()
    })?;
    Ok(DeviceSpan { start, end, name })
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn reject_overlapping_gdn_step_spans(left: DeviceSpan, right: DeviceSpan) -> Result<()> {
    if left.start < right.end && right.start < left.end {
        unsupported_gdn_step_shape(format!(
            "writable {} span aliases {} span",
            left.name, right.name
        ))
    } else {
        Ok(())
    }
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn gdn_step_u32(name: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: GDN_GROUPED_STEP_KERNEL,
            msg: format!("{name} {value} exceeds the HIP ABI u32 domain"),
        }
        .build()
    })
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn unsupported_gdn_step_shape<T>(msg: String) -> Result<T> {
    UnsupportedShapeSnafu {
        kernel: GDN_GROUPED_STEP_KERNEL,
        msg,
    }
    .fail()
}

fn checked_product(left: usize, right: usize, dimensions: &'static str) -> GdnResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| DimensionProductOverflowSnafu { dimensions }.build())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecurrentAllocationPlan {
    query_and_key: usize,
    output: usize,
    state: usize,
    state_times_key: usize,
    delta: usize,
}

impl RecurrentAllocationPlan {
    fn try_from_dimensions(
        token_count: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> GdnResult<Self> {
        validate_nonzero_dimension("key_dim", key_dim)?;
        validate_nonzero_dimension("value_dim", value_dim)?;
        validate_nonzero_dimension("token_count", token_count)?;
        Ok(Self {
            query_and_key: checked_product(token_count, key_dim, "token_count * key_dim")?,
            output: checked_product(token_count, value_dim, "token_count * value_dim")?,
            state: checked_product(key_dim, value_dim, "key_dim * value_dim")?,
            state_times_key: value_dim,
            delta: value_dim,
        })
    }
}

fn checked_allocation_sum(sum: usize, elements: usize) -> GdnResult<usize> {
    sum.checked_add(elements)
        .ok_or_else(|| DimensionSumOverflowSnafu.build())
}

fn reserve_f32(allocation: &'static str, elements: usize) -> GdnResult<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .context(AllocationSnafu {
            allocation,
            elements,
        })?;
    Ok(values)
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
    fn recurrence_state_crosses_every_nonempty_chunk_boundary() -> GdnResult<()> {
        let q = [0.25, -0.5, 1.25, 0.75, -1.0, 0.125];
        let k = [0.5, 1.0, -0.75, 0.25, 1.5, -0.5];
        let v = [1.0, -2.0, 0.5, -0.25, 1.25, 2.0, 0.75, -1.5, 0.125];
        let beta = [0.25, 0.75, 0.5];
        let g = [-0.125, 0.0625, -0.25];
        let state = [0.25, -0.5, 1.0, -1.25, 0.75, 0.125];
        let full_input =
            RecurrentInput::new(&q, &k, &v, &beta, &g, SCALE, &state, KEY_DIM, VALUE_DIM)?;
        let full = recurrent_fwd(&full_input)?;

        for first_token_count in 1..beta.len() {
            let first_key_elements = first_token_count * KEY_DIM;
            let first_value_elements = first_token_count * VALUE_DIM;
            let first_input = RecurrentInput::new(
                &q[..first_key_elements],
                &k[..first_key_elements],
                &v[..first_value_elements],
                &beta[..first_token_count],
                &g[..first_token_count],
                SCALE,
                &state,
                KEY_DIM,
                VALUE_DIM,
            )?;
            let first = recurrent_fwd(&first_input)?;
            let rest_input = RecurrentInput::new(
                &q[first_key_elements..],
                &k[first_key_elements..],
                &v[first_value_elements..],
                &beta[first_token_count..],
                &g[first_token_count..],
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
                "split at token {first_token_count} must preserve every output"
            );
            assert_eq!(
                full.state(),
                rest.state(),
                "split at token {first_token_count} must preserve final state"
            );
        }
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
    const THREE_TO_ONE_KEY_HEAD_COUNT: usize = 1;
    const THREE_TO_ONE_VALUE_HEAD_COUNT: usize = 3;
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
        for (key_head_count, value_head_count) in [
            (KEY_HEAD_COUNT, VALUE_HEAD_COUNT),
            (THREE_TO_ONE_KEY_HEAD_COUNT, THREE_TO_ONE_VALUE_HEAD_COUNT),
        ] {
            let fixture = multi_head_fixture(key_head_count, value_head_count);
            let input = multi_head_input(
                &fixture.q,
                &fixture.k,
                &fixture.v,
                &fixture.beta,
                &fixture.g,
                &fixture.state,
                TOKEN_COUNT,
                key_head_count,
                value_head_count,
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
                key_head_count,
                value_head_count,
                KEY_DIM,
                MULTI_HEAD_VALUE_DIM,
            );

            assert_close(actual.output(), &expected_output, "multi-head output");
            assert_close(actual.state(), &expected_state, "multi-head state");
        }
        Ok(())
    }

    #[test]
    fn one_token_grouped_and_pre_tiled_profiles_match_independent_f64_oracle() -> GdnResult<()> {
        for (key_head_count, value_head_count) in [(KEY_HEAD_COUNT, VALUE_HEAD_COUNT), (2, 2)] {
            let fixture = multi_head_fixture(key_head_count, value_head_count);
            let q = head_major_token_window(&fixture.q, key_head_count, TOKEN_COUNT, KEY_DIM, 0, 1);
            let k = head_major_token_window(&fixture.k, key_head_count, TOKEN_COUNT, KEY_DIM, 0, 1);
            let v = head_major_token_window(
                &fixture.v,
                value_head_count,
                TOKEN_COUNT,
                MULTI_HEAD_VALUE_DIM,
                0,
                1,
            );
            let beta =
                head_major_token_window(&fixture.beta, value_head_count, TOKEN_COUNT, 1, 0, 1);
            let g = head_major_token_window(&fixture.g, value_head_count, TOKEN_COUNT, 1, 0, 1);
            let input = multi_head_input(
                &q,
                &k,
                &v,
                &beta,
                &g,
                &fixture.state,
                1,
                key_head_count,
                value_head_count,
            )?;
            let actual = multi_head_recurrent_fwd(&input)?;
            let (expected_output, expected_state) = oracle_multi_head_recurrence(
                &q,
                &k,
                &v,
                &beta,
                &g,
                SCALE,
                &fixture.state,
                1,
                key_head_count,
                value_head_count,
                KEY_DIM,
                MULTI_HEAD_VALUE_DIM,
            );

            assert_close(actual.output(), &expected_output, "one-token output");
            assert_close(actual.state(), &expected_state, "one-token next state");
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    struct ValidGdnStepBuffers {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        beta: Vec<f32>,
        g: Vec<f32>,
        state_in: Vec<f32>,
        state_out: Vec<f32>,
        output: Vec<f32>,
    }

    #[cfg(feature = "gpu")]
    impl ValidGdnStepBuffers {
        fn from_plan(plan: MultiHeadRecurrentAllocationPlan) -> Self {
            Self {
                q: vec![1.0_f32; plan.query_and_key_elements()],
                k: vec![1.0_f32; plan.query_and_key_elements()],
                v: vec![1.0_f32; plan.output_elements()],
                beta: vec![1.0_f32; plan.scalar_elements()],
                g: vec![0.0_f32; plan.scalar_elements()],
                state_in: vec![0.0_f32; plan.state_elements()],
                state_out: vec![0.0_f32; plan.state_elements()],
                output: vec![0.0_f32; plan.output_elements()],
            }
        }

        fn validate(
            &mut self,
            plan: MultiHeadRecurrentAllocationPlan,
            scale: f32,
        ) -> Result<GdnStepAbi> {
            validate_gdn_step_launch(
                plan,
                self.q.as_ptr(),
                self.q.len(),
                self.k.as_ptr(),
                self.k.len(),
                self.v.as_ptr(),
                self.v.len(),
                self.beta.as_ptr(),
                self.beta.len(),
                self.g.as_ptr(),
                self.g.len(),
                scale,
                self.state_in.as_ptr(),
                self.state_in.len(),
                self.state_out.as_mut_ptr(),
                self.state_out.len(),
                self.output.as_mut_ptr(),
                self.output.len(),
            )
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn staged_gpu_step_refuses_unsupported_shape_alias_and_overflow()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let two_token_plan = MultiHeadRecurrentAllocationPlan::try_from_dimensions(2, 1, 1, 1, 1)?;
        let mut two_token_buffers = ValidGdnStepBuffers::from_plan(two_token_plan);
        assert!(matches!(
            two_token_buffers.validate(two_token_plan, 1.0),
            Err(crate::Error::UnsupportedShape { .. })
        ));

        let plan = MultiHeadRecurrentAllocationPlan::try_from_dimensions(1, 1, 1, 1, 1)?;
        let q = [1.0_f32];
        let k = [1.0_f32];
        let v = [1.0_f32];
        let beta = [1.0_f32];
        let g = [0.0_f32];
        let state = [0.0_f32];
        let mut output = [0.0_f32];
        let grouped_plan = MultiHeadRecurrentAllocationPlan::try_from_dimensions(1, 2, 4, 3, 2)?;
        let mut grouped_buffers = ValidGdnStepBuffers::from_plan(grouped_plan);
        let abi = grouped_buffers.validate(grouped_plan, 1.0)?;
        assert_eq!(
            (
                abi.key_head_count,
                abi.value_head_count,
                abi.key_dim,
                abi.value_dim,
            ),
            (2, 4, 3, 2),
            "the checked ABI must preserve the allocation owner's grouped geometry"
        );
        let too_wide_plan = MultiHeadRecurrentAllocationPlan::try_from_dimensions(
            1,
            1,
            1,
            1,
            MAX_GDN_VALUE_DIM + 1,
        )?;
        let mut too_wide_buffers = ValidGdnStepBuffers::from_plan(too_wide_plan);
        assert!(matches!(
            too_wide_buffers.validate(too_wide_plan, 1.0),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        for invalid_scale in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
            -f32::MIN_POSITIVE / 2.0,
        ] {
            assert!(matches!(
                grouped_buffers.validate(grouped_plan, invalid_scale),
                Err(crate::Error::UnsupportedShape { .. })
            ));
        }
        assert!(matches!(
            validate_gdn_step_launch(
                plan,
                q.as_ptr(),
                0,
                k.as_ptr(),
                k.len(),
                v.as_ptr(),
                v.len(),
                beta.as_ptr(),
                beta.len(),
                g.as_ptr(),
                g.len(),
                1.0,
                state.as_ptr(),
                state.len(),
                output.as_mut_ptr(),
                output.len(),
                output.as_mut_ptr(),
                output.len(),
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        assert!(matches!(
            validate_gdn_step_launch(
                plan,
                q.as_ptr(),
                q.len(),
                k.as_ptr(),
                k.len(),
                v.as_ptr(),
                v.len(),
                beta.as_ptr(),
                beta.len(),
                g.as_ptr(),
                g.len(),
                1.0,
                state.as_ptr(),
                state.len(),
                state.as_ptr().cast_mut(),
                state.len(),
                output.as_mut_ptr(),
                output.len(),
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        assert!(matches!(
            checked_device_span(
                core::ptr::NonNull::<f32>::dangling().as_ptr(),
                usize::MAX,
                "overflow"
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        if let Ok(abi_overflow) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(matches!(
                gdn_step_u32("abi overflow", abi_overflow),
                Err(crate::Error::UnsupportedShape { .. })
            ));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
    fn reserved_device_grouped_step_matches_oracle_for_grouped_equal_and_continuation()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer, Stream};

        let device = Device::new(0).map_err(|error| format!("open reserved device 0: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;

        for (key_head_count, value_head_count) in [(KEY_HEAD_COUNT, VALUE_HEAD_COUNT), (2, 2)] {
            let fixture = multi_head_fixture(key_head_count, value_head_count);
            let first = grouped_token_inputs(&fixture, key_head_count, value_head_count, 0);
            let second = grouped_token_inputs(&fixture, key_head_count, value_head_count, 1);
            let plan = MultiHeadRecurrentAllocationPlan::try_from_dimensions(
                1,
                key_head_count,
                value_head_count,
                KEY_DIM,
                MULTI_HEAD_VALUE_DIM,
            )
            .map_err(|error| format!("build one-token plan: {error}"))?;
            let state_in = DeviceBuffer::<f32>::from_host(&device, &fixture.state)
                .map_err(|error| format!("upload initial state: {error}"))?;

            let (first_output, first_state) =
                launch_reserved_device_step(&device, &stream, plan, &first, &state_in)?;
            let (second_output, second_state) =
                launch_reserved_device_step(&device, &stream, plan, &second, &first_state)?;
            let actual_first_output = read_reserved_device_buffer(&first_output)?;
            let actual_first_state = read_reserved_device_buffer(&first_state)?;
            let actual_second_output = read_reserved_device_buffer(&second_output)?;
            let actual_second_state = read_reserved_device_buffer(&second_state)?;
            let (expected_first_output, expected_first_state) = oracle_multi_head_recurrence(
                &first.q,
                &first.k,
                &first.v,
                &first.beta,
                &first.g,
                SCALE,
                &fixture.state,
                1,
                key_head_count,
                value_head_count,
                KEY_DIM,
                MULTI_HEAD_VALUE_DIM,
            );
            let q = head_major_token_window(&fixture.q, key_head_count, TOKEN_COUNT, KEY_DIM, 0, 2);
            let k = head_major_token_window(&fixture.k, key_head_count, TOKEN_COUNT, KEY_DIM, 0, 2);
            let v = head_major_token_window(
                &fixture.v,
                value_head_count,
                TOKEN_COUNT,
                MULTI_HEAD_VALUE_DIM,
                0,
                2,
            );
            let beta =
                head_major_token_window(&fixture.beta, value_head_count, TOKEN_COUNT, 1, 0, 2);
            let g = head_major_token_window(&fixture.g, value_head_count, TOKEN_COUNT, 1, 0, 2);
            let (expected_full_output, expected_full_state) = oracle_multi_head_recurrence(
                &q,
                &k,
                &v,
                &beta,
                &g,
                SCALE,
                &fixture.state,
                2,
                key_head_count,
                value_head_count,
                KEY_DIM,
                MULTI_HEAD_VALUE_DIM,
            );

            assert_close(
                &actual_first_output,
                &expected_first_output,
                "device one-token output",
            );
            assert_close(
                &actual_first_state,
                &expected_first_state,
                "device one-token next state",
            );
            let actual_full_output = join_head_major_outputs(
                &actual_first_output,
                &actual_second_output,
                value_head_count,
                1,
                1,
                MULTI_HEAD_VALUE_DIM,
            );
            assert_close(
                &actual_full_output,
                &expected_full_output,
                "device continuation output",
            );
            assert_close(
                &actual_second_state,
                &expected_full_state,
                "device continuation next state",
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
    #[test]
    fn staged_gpu_step_cpu_only_witness_never_initializes_hip() {
        assert!(matches!(
            no_gpu_gdn_step_refusal(),
            Err(crate::Error::NoGpuBuild { .. })
        ));
    }

    #[test]
    fn multi_head_recurrence_state_crosses_every_nonempty_chunk_boundary() -> GdnResult<()> {
        for (key_head_count, value_head_count) in [
            (KEY_HEAD_COUNT, VALUE_HEAD_COUNT),
            (THREE_TO_ONE_KEY_HEAD_COUNT, THREE_TO_ONE_VALUE_HEAD_COUNT),
        ] {
            let fixture = multi_head_fixture(key_head_count, value_head_count);
            let full_input = multi_head_input(
                &fixture.q,
                &fixture.k,
                &fixture.v,
                &fixture.beta,
                &fixture.g,
                &fixture.state,
                TOKEN_COUNT,
                key_head_count,
                value_head_count,
            )?;
            let full = multi_head_recurrent_fwd(&full_input)?;

            for first_token_count in 1..TOKEN_COUNT {
                let first_q = head_major_token_window(
                    &fixture.q,
                    key_head_count,
                    TOKEN_COUNT,
                    KEY_DIM,
                    0,
                    first_token_count,
                );
                let first_k = head_major_token_window(
                    &fixture.k,
                    key_head_count,
                    TOKEN_COUNT,
                    KEY_DIM,
                    0,
                    first_token_count,
                );
                let first_v = head_major_token_window(
                    &fixture.v,
                    value_head_count,
                    TOKEN_COUNT,
                    MULTI_HEAD_VALUE_DIM,
                    0,
                    first_token_count,
                );
                let first_beta = head_major_token_window(
                    &fixture.beta,
                    value_head_count,
                    TOKEN_COUNT,
                    1,
                    0,
                    first_token_count,
                );
                let first_g = head_major_token_window(
                    &fixture.g,
                    value_head_count,
                    TOKEN_COUNT,
                    1,
                    0,
                    first_token_count,
                );
                let first_input = multi_head_input(
                    &first_q,
                    &first_k,
                    &first_v,
                    &first_beta,
                    &first_g,
                    &fixture.state,
                    first_token_count,
                    key_head_count,
                    value_head_count,
                )?;
                let first = multi_head_recurrent_fwd(&first_input)?;

                let remaining_token_count = TOKEN_COUNT - first_token_count;
                let rest_q = head_major_token_window(
                    &fixture.q,
                    key_head_count,
                    TOKEN_COUNT,
                    KEY_DIM,
                    first_token_count,
                    TOKEN_COUNT,
                );
                let rest_k = head_major_token_window(
                    &fixture.k,
                    key_head_count,
                    TOKEN_COUNT,
                    KEY_DIM,
                    first_token_count,
                    TOKEN_COUNT,
                );
                let rest_v = head_major_token_window(
                    &fixture.v,
                    value_head_count,
                    TOKEN_COUNT,
                    MULTI_HEAD_VALUE_DIM,
                    first_token_count,
                    TOKEN_COUNT,
                );
                let rest_beta = head_major_token_window(
                    &fixture.beta,
                    value_head_count,
                    TOKEN_COUNT,
                    1,
                    first_token_count,
                    TOKEN_COUNT,
                );
                let rest_g = head_major_token_window(
                    &fixture.g,
                    value_head_count,
                    TOKEN_COUNT,
                    1,
                    first_token_count,
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
                    key_head_count,
                    value_head_count,
                )?;
                let rest = multi_head_recurrent_fwd(&rest_input)?;
                let joined_output = join_head_major_outputs(
                    first.output(),
                    rest.output(),
                    value_head_count,
                    first_token_count,
                    remaining_token_count,
                    MULTI_HEAD_VALUE_DIM,
                );

                assert_eq!(
                    full.output(),
                    joined_output,
                    "split at token {first_token_count} must preserve every multi-head output"
                );
                assert_eq!(
                    full.state(),
                    rest.state(),
                    "split at token {first_token_count} must preserve final multi-head state"
                );
            }
        }
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
        let fixture = multi_head_fixture(KEY_HEAD_COUNT, VALUE_HEAD_COUNT);
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

    fn multi_head_fixture(key_head_count: usize, value_head_count: usize) -> MultiHeadFixture {
        MultiHeadFixture {
            q: fixture_values(key_head_count * TOKEN_COUNT * KEY_DIM, -0.5),
            k: fixture_values(key_head_count * TOKEN_COUNT * KEY_DIM, 0.25),
            v: fixture_values(value_head_count * TOKEN_COUNT * MULTI_HEAD_VALUE_DIM, -0.75),
            beta: beta_values(value_head_count * TOKEN_COUNT),
            g: gate_values(value_head_count * TOKEN_COUNT),
            state: fixture_values(value_head_count * KEY_DIM * MULTI_HEAD_VALUE_DIM, 0.125),
        }
    }

    struct GroupedTokenInputs {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        beta: Vec<f32>,
        g: Vec<f32>,
    }

    fn grouped_token_inputs(
        fixture: &MultiHeadFixture,
        key_head_count: usize,
        value_head_count: usize,
        token_index: usize,
    ) -> GroupedTokenInputs {
        let end_token = token_index + 1;
        GroupedTokenInputs {
            q: head_major_token_window(
                &fixture.q,
                key_head_count,
                TOKEN_COUNT,
                KEY_DIM,
                token_index,
                end_token,
            ),
            k: head_major_token_window(
                &fixture.k,
                key_head_count,
                TOKEN_COUNT,
                KEY_DIM,
                token_index,
                end_token,
            ),
            v: head_major_token_window(
                &fixture.v,
                value_head_count,
                TOKEN_COUNT,
                MULTI_HEAD_VALUE_DIM,
                token_index,
                end_token,
            ),
            beta: head_major_token_window(
                &fixture.beta,
                value_head_count,
                TOKEN_COUNT,
                1,
                token_index,
                end_token,
            ),
            g: head_major_token_window(
                &fixture.g,
                value_head_count,
                TOKEN_COUNT,
                1,
                token_index,
                end_token,
            ),
        }
    }

    #[cfg(feature = "gpu")]
    fn launch_reserved_device_step(
        device: &hipcore::Device,
        stream: &hipcore::Stream,
        plan: MultiHeadRecurrentAllocationPlan,
        input: &GroupedTokenInputs,
        state_in: &hipcore::DeviceBuffer<f32>,
    ) -> core::result::Result<(hipcore::DeviceBuffer<f32>, hipcore::DeviceBuffer<f32>), String>
    {
        let q = hipcore::DeviceBuffer::<f32>::from_host(device, &input.q)
            .map_err(|error| format!("upload q: {error}"))?;
        let k = hipcore::DeviceBuffer::<f32>::from_host(device, &input.k)
            .map_err(|error| format!("upload k: {error}"))?;
        let v = hipcore::DeviceBuffer::<f32>::from_host(device, &input.v)
            .map_err(|error| format!("upload v: {error}"))?;
        let beta = hipcore::DeviceBuffer::<f32>::from_host(device, &input.beta)
            .map_err(|error| format!("upload beta: {error}"))?;
        let g = hipcore::DeviceBuffer::<f32>::from_host(device, &input.g)
            .map_err(|error| format!("upload g: {error}"))?;
        let state_out = hipcore::DeviceBuffer::<f32>::alloc(device, plan.state_elements())
            .map_err(|error| format!("allocate next state: {error}"))?;
        let output = hipcore::DeviceBuffer::<f32>::alloc(device, plan.output_elements())
            .map_err(|error| format!("allocate output: {error}"))?;

        // SAFETY: each typed device buffer has the allocation-plan-derived
        // extent, stays live through synchronization, and writable buffers
        // are distinct from every immutable input.
        unsafe {
            launch_multi_head_recurrent_step_f32(
                plan,
                q.as_device_ptr(),
                q.len(),
                k.as_device_ptr(),
                k.len(),
                v.as_device_ptr(),
                v.len(),
                beta.as_device_ptr(),
                beta.len(),
                g.as_device_ptr(),
                g.len(),
                SCALE,
                state_in.as_device_ptr(),
                state_in.len(),
                state_out.as_device_ptr(),
                state_out.len(),
                output.as_device_ptr(),
                output.len(),
                stream,
            )
            .map_err(|error| format!("launch grouped GDN step: {error}"))?;
        }
        stream
            .synchronize()
            .map_err(|error| format!("synchronize grouped GDN step: {error}"))?;
        Ok((output, state_out))
    }

    #[cfg(feature = "gpu")]
    fn read_reserved_device_buffer(
        buffer: &hipcore::DeviceBuffer<f32>,
    ) -> core::result::Result<Vec<f32>, String> {
        let mut host = vec![0.0_f32; buffer.len()];
        buffer
            .copy_to_host(&mut host)
            .map_err(|error| format!("read staged device result: {error}"))?;
        Ok(host)
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
