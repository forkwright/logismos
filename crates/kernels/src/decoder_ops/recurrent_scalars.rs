//! Checked T=1 recurrent beta and log-decay scalar operation.

#[cfg(not(logismos_no_gpu_kernels))]
use std::ffi::c_void;

use hipcore::Stream;

use super::{ELEMENTWISE_THREADS, Result};

const RECURRENT_SCALARS_KERNEL: &str = "decoder_recurrent_scalars_f32";

#[cfg(not(logismos_no_gpu_kernels))]
unsafe extern "C" {
    fn logismos_launch_recurrent_scalars_f32(
        alpha_f32: *const c_void,
        dt_f32: *const c_void,
        a_f32: *const c_void,
        beta_projection_f32: *const c_void,
        beta_f32: *mut c_void,
        log_decay_f32: *mut c_void,
        value_heads: u32,
        stream: *mut c_void,
    ) -> u32;
}

/// Checked T=1 value-head geometry for Qwen3.5 recurrent scalars.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentScalarsF32Plan {
    value_heads: usize,
    value_heads_u32: u32,
}

impl RecurrentScalarsF32Plan {
    /// Admit one nonempty T=1 vector of recurrent value-head scalars.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when `value_heads` is zero,
    /// cannot form the exact f32 operand/output layout, cannot fit the HIP ABI,
    /// or cannot fit the fixed elementwise launch grid.
    pub fn try_from_value_heads(value_heads: usize) -> Result<Self> {
        super::validate_nonzero(RECURRENT_SCALARS_KERNEL, "value_heads", value_heads)?;
        super::validate_f32_layout(
            RECURRENT_SCALARS_KERNEL,
            "recurrent scalar operands and outputs",
            value_heads,
        )?;
        super::validate_grid(RECURRENT_SCALARS_KERNEL, value_heads, ELEMENTWISE_THREADS)?;
        Ok(Self {
            value_heads,
            value_heads_u32: super::abi_u32(RECURRENT_SCALARS_KERNEL, "value_heads", value_heads)?,
        })
    }

    /// Return the exact T=1 scalar count and value-head count.
    #[must_use]
    pub const fn value_heads(self) -> usize {
        self.value_heads
    }
}

/// Launch Qwen3.5's T=1 recurrent beta and log-decay scalar transforms.
///
/// Each value head uses this f32 operation order:
/// `beta = sigmoid(beta_projection)`; `sum = alpha + dt`;
/// `log_decay = a * softplus(sum)`. The sigmoid and softplus use their stable
/// CPU-equivalent branches. A negative-infinite `sum` intentionally saturates
/// softplus to zero; positive infinity and NaN are outside the numerical domain.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] for a mismatched exact span or
/// overlap; [`crate::Error::NoGpuBuild`] for a CPU-only build; and propagated
/// stream-current or HIP launch failures.
///
/// # Safety
///
/// Every pointer must identify a correctly aligned allocation on `stream`'s
/// device for exactly `plan.value_heads()` f32 values and remain live through
/// completion. All inputs must remain immutable; both outputs must remain
/// exclusively writable and must not alias each other or any input. Inputs and
/// outputs must be finite normal-or-zero, except that finite `alpha + dt` may
/// overflow only to negative infinity for the documented softplus saturation.
#[expect(
    clippy::too_many_arguments,
    reason = "six exact scalar spans are one non-aliased recurrent operation contract"
)]
pub unsafe fn launch_recurrent_scalars_f32(
    plan: RecurrentScalarsF32Plan,
    alpha_f32: *const f32,
    alpha_elements: usize,
    dt_f32: *const f32,
    dt_elements: usize,
    a_f32: *const f32,
    a_elements: usize,
    beta_projection_f32: *const f32,
    beta_projection_elements: usize,
    beta_f32: *mut f32,
    beta_elements: usize,
    log_decay_f32: *mut f32,
    log_decay_elements: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            alpha_f32,
            alpha_elements,
            dt_f32,
            dt_elements,
            a_f32,
            a_elements,
            beta_projection_f32,
            beta_projection_elements,
            beta_f32,
            beta_elements,
            log_decay_f32,
            log_decay_elements,
            stream,
        );
        super::no_gpu_refusal(RECURRENT_SCALARS_KERNEL)
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_recurrent_scalars_launch(
            plan,
            alpha_f32,
            alpha_elements,
            dt_f32,
            dt_elements,
            a_f32,
            a_elements,
            beta_projection_f32,
            beta_projection_elements,
            beta_f32,
            beta_elements,
            log_decay_f32,
            log_decay_elements,
        )?;
        stream.make_current()?;
        // SAFETY: checked exact spans plus the caller's numerical and ownership
        // contract establish the private kernel ABI's preconditions.
        let code = unsafe {
            logismos_launch_recurrent_scalars_f32(
                alpha_f32.cast::<c_void>(),
                dt_f32.cast::<c_void>(),
                a_f32.cast::<c_void>(),
                beta_projection_f32.cast::<c_void>(),
                beta_f32.cast::<c_void>(),
                log_decay_f32.cast::<c_void>(),
                plan.value_heads_u32,
                stream.raw().cast::<c_void>(),
            )
        };
        super::launch_result(RECURRENT_SCALARS_KERNEL, code)
    }
}

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
#[expect(
    clippy::too_many_arguments,
    reason = "six exact scalar spans are one non-aliased recurrent operation contract"
)]
fn validate_recurrent_scalars_launch(
    plan: RecurrentScalarsF32Plan,
    alpha_f32: *const f32,
    alpha_elements: usize,
    dt_f32: *const f32,
    dt_elements: usize,
    a_f32: *const f32,
    a_elements: usize,
    beta_projection_f32: *const f32,
    beta_projection_elements: usize,
    beta_f32: *mut f32,
    beta_elements: usize,
    log_decay_f32: *mut f32,
    log_decay_elements: usize,
) -> Result<()> {
    let expected = plan.value_heads;
    super::validate_length(RECURRENT_SCALARS_KERNEL, "alpha", alpha_elements, expected)?;
    super::validate_length(RECURRENT_SCALARS_KERNEL, "dt", dt_elements, expected)?;
    super::validate_length(RECURRENT_SCALARS_KERNEL, "a", a_elements, expected)?;
    super::validate_length(
        RECURRENT_SCALARS_KERNEL,
        "beta projection",
        beta_projection_elements,
        expected,
    )?;
    super::validate_length(
        RECURRENT_SCALARS_KERNEL,
        "beta output",
        beta_elements,
        expected,
    )?;
    super::validate_length(
        RECURRENT_SCALARS_KERNEL,
        "log-decay output",
        log_decay_elements,
        expected,
    )?;

    let alpha = super::checked_f32_device_span(
        RECURRENT_SCALARS_KERNEL,
        alpha_f32,
        alpha_elements,
        "alpha",
    )?;
    let dt = super::checked_f32_device_span(RECURRENT_SCALARS_KERNEL, dt_f32, dt_elements, "dt")?;
    let a = super::checked_f32_device_span(RECURRENT_SCALARS_KERNEL, a_f32, a_elements, "a")?;
    let beta_projection = super::checked_f32_device_span(
        RECURRENT_SCALARS_KERNEL,
        beta_projection_f32,
        beta_projection_elements,
        "beta projection",
    )?;
    let beta = super::checked_f32_device_span(
        RECURRENT_SCALARS_KERNEL,
        beta_f32.cast_const(),
        beta_elements,
        "beta output",
    )?;
    let log_decay = super::checked_f32_device_span(
        RECURRENT_SCALARS_KERNEL,
        log_decay_f32.cast_const(),
        log_decay_elements,
        "log-decay output",
    )?;
    for input in [alpha, dt, a, beta_projection] {
        super::reject_overlapping_f32_spans(RECURRENT_SCALARS_KERNEL, beta, input)?;
        super::reject_overlapping_f32_spans(RECURRENT_SCALARS_KERNEL, log_decay, input)?;
    }
    super::reject_overlapping_f32_spans(RECURRENT_SCALARS_KERNEL, beta, log_decay)
}

#[cfg(test)]
fn recurrent_scalars_native_order_reference(
    plan: RecurrentScalarsF32Plan,
    alpha: &[f32],
    dt: &[f32],
    a: &[f32],
    beta_projection: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expected = plan.value_heads;
    for (name, values) in [
        ("alpha", alpha),
        ("dt", dt),
        ("a", a),
        ("beta projection", beta_projection),
    ] {
        super::validate_reference_length(RECURRENT_SCALARS_KERNEL, name, values.len(), expected)?;
    }
    let mut beta = super::reserve_native_reference("recurrent beta reference", expected)?;
    let mut log_decay = super::reserve_native_reference("recurrent log-decay reference", expected)?;
    for (((alpha_value, dt_value), a_value), beta_value) in alpha
        .iter()
        .zip(dt.iter())
        .zip(a.iter())
        .zip(beta_projection.iter())
    {
        beta.push(stable_sigmoid_f32(*beta_value));
        let alpha_plus_dt = *alpha_value + *dt_value;
        log_decay.push(*a_value * stable_softplus_f32(alpha_plus_dt));
    }
    Ok((beta, log_decay))
}

#[cfg(test)]
fn stable_sigmoid_f32(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
fn stable_softplus_f32(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-3;

    #[test]
    fn native_order_tracks_independent_f64_oracle_and_input_mutations()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = RecurrentScalarsF32Plan::try_from_value_heads(7)?;
        let alpha = [-50.0_f32, -5.0, -0.5, 0.0, 2.0, 8.0, 50.0];
        let dt = [1.0_f32, 0.25, 1.0, -2.0, 0.5, -3.0, -1.0];
        let a = [-1.0_f32, -0.5, -0.25, 0.5, 1.5, -2.0, 0.25];
        let beta_projection = [-80.0_f32, -8.0, -0.75, 0.0, 1.25, 6.0, 80.0];
        let (beta, log_decay) =
            recurrent_scalars_native_order_reference(plan, &alpha, &dt, &a, &beta_projection)?;
        let (expected_beta, expected_log_decay) =
            recurrent_scalars_f64_oracle(&alpha, &dt, &a, &beta_projection);
        assert_close_f64(&beta, &expected_beta, "recurrent beta");
        assert_close_f64(&log_decay, &expected_log_decay, "recurrent log decay");

        let mut beta_projection_mutated = beta_projection;
        beta_projection_mutated[3] += 3.0;
        let (mutated_beta, unchanged_log_decay) = recurrent_scalars_native_order_reference(
            plan,
            &alpha,
            &dt,
            &a,
            &beta_projection_mutated,
        )?;
        assert_ne!(beta, mutated_beta, "beta must depend on its projection");
        assert_eq!(
            log_decay, unchanged_log_decay,
            "beta projection must not affect log decay"
        );

        let mut alpha_mutated = alpha;
        alpha_mutated[1] += 2.0;
        let (_, mutated_log_decay) = recurrent_scalars_native_order_reference(
            plan,
            &alpha_mutated,
            &dt,
            &a,
            &beta_projection,
        )?;
        assert_ne!(
            log_decay, mutated_log_decay,
            "log decay must preserve alpha-plus-dt dependence"
        );

        let mut a_mutated = a;
        a_mutated[4] *= -0.5;
        let (_, scale_mutated_log_decay) = recurrent_scalars_native_order_reference(
            plan,
            &alpha,
            &dt,
            &a_mutated,
            &beta_projection,
        )?;
        assert_ne!(
            log_decay, scale_mutated_log_decay,
            "log decay must preserve the per-head A scale"
        );
        Ok(())
    }

    #[test]
    fn negative_overflow_sum_preserves_cpu_softplus_saturation()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = RecurrentScalarsF32Plan::try_from_value_heads(1)?;
        let (beta, log_decay) = recurrent_scalars_native_order_reference(
            plan,
            &[-f32::MAX],
            &[-f32::MAX],
            &[-1.0],
            &[0.0],
        )?;
        assert_eq!(beta, vec![0.5]);
        assert_eq!(log_decay, vec![0.0]);
        Ok(())
    }

    #[test]
    fn plan_and_span_validation_refuse_invalid_shapes_and_output_aliases()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        assert!(RecurrentScalarsF32Plan::try_from_value_heads(0).is_err());
        if let Ok(too_wide) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(RecurrentScalarsF32Plan::try_from_value_heads(too_wide).is_err());
        }

        let plan = RecurrentScalarsF32Plan::try_from_value_heads(2)?;
        let alpha = [0.0_f32; 2];
        let dt = [0.0_f32; 2];
        let a = [1.0_f32; 2];
        let beta_projection = [0.0_f32; 2];
        let mut beta = [0.0_f32; 2];
        let mut log_decay = [0.0_f32; 2];
        validate_recurrent_scalars_launch(
            plan,
            alpha.as_ptr(),
            alpha.len(),
            dt.as_ptr(),
            dt.len(),
            a.as_ptr(),
            a.len(),
            beta_projection.as_ptr(),
            beta_projection.len(),
            beta.as_mut_ptr(),
            beta.len(),
            log_decay.as_mut_ptr(),
            log_decay.len(),
        )?;
        assert!(
            validate_recurrent_scalars_launch(
                plan,
                alpha.as_ptr(),
                alpha.len() - 1,
                dt.as_ptr(),
                dt.len(),
                a.as_ptr(),
                a.len(),
                beta_projection.as_ptr(),
                beta_projection.len(),
                beta.as_mut_ptr(),
                beta.len(),
                log_decay.as_mut_ptr(),
                log_decay.len(),
            )
            .is_err()
        );
        assert!(
            validate_recurrent_scalars_launch(
                plan,
                alpha.as_ptr(),
                alpha.len(),
                dt.as_ptr(),
                dt.len(),
                a.as_ptr(),
                a.len(),
                beta_projection.as_ptr(),
                beta_projection.len(),
                beta.as_mut_ptr(),
                beta.len(),
                beta.as_mut_ptr(),
                beta.len(),
            )
            .is_err()
        );
        Ok(())
    }

    fn recurrent_scalars_f64_oracle(
        alpha: &[f32],
        dt: &[f32],
        a: &[f32],
        beta_projection: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let beta = beta_projection
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                if value >= 0.0 {
                    1.0 / (1.0 + (-value).exp())
                } else {
                    let exponential = value.exp();
                    exponential / (1.0 + exponential)
                }
            })
            .collect();
        let log_decay = alpha
            .iter()
            .zip(dt.iter())
            .zip(a.iter())
            .map(|((alpha, dt), a)| {
                let sum = f64::from(*alpha) + f64::from(*dt);
                let softplus = sum.max(0.0) + (-sum.abs()).exp().ln_1p();
                f64::from(*a) * softplus
            })
            .collect();
        (beta, log_decay)
    }

    fn assert_close_f64(actual: &[f32], expected: &[f64], operation: &str) {
        assert_eq!(actual.len(), expected.len(), "{operation} output length");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f64::from(*actual) - expected).abs() <= f64::from(TOLERANCE),
                "{operation} index {index}: got {actual}, expected {expected}"
            );
        }
    }
}
