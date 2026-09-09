//! Checked T=1 recurrent beta and log-decay scalar operation.

use core::ffi::c_void;

use hipcore::Stream;

use super::{ELEMENTWISE_THREADS, Result};
use crate::numerical_status::NativeNumericalStatus;

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
        numerical_status: *mut c_void,
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
/// branches, but this native operation makes no CPU transcendental bit-parity
/// claim.
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
/// every operand, intermediate, and output must be finite normal-or-zero
/// through completion.
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
                core::ptr::null_mut(),
                stream.raw().cast::<c_void>(),
            )
        };
        super::launch_result(RECURRENT_SCALARS_KERNEL, code)
    }
}

/// Launch recurrent scalars while recording explicit numerical-domain failures.
///
/// # Safety
///
/// The raw launcher's pointer, lifetime, ownership, and stream requirements
/// apply. `status` must remain live through stream completion on the same device.
#[expect(
    clippy::too_many_arguments,
    reason = "six exact scalar spans are one non-aliased recurrent operation contract"
)]
pub unsafe fn launch_recurrent_scalars_f32_checked(
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
    status: &NativeNumericalStatus,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = status;
        // SAFETY: this forwards the unchanged raw arguments solely to retain its typed CPU refusal.
        unsafe {
            launch_recurrent_scalars_f32(
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
            )
        }
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
        // SAFETY: checked spans and the caller's status lifetime contract establish this private ABI.
        let code = unsafe {
            logismos_launch_recurrent_scalars_f32(
                alpha_f32.cast::<c_void>(),
                dt_f32.cast::<c_void>(),
                a_f32.cast::<c_void>(),
                beta_projection_f32.cast::<c_void>(),
                beta_f32.cast::<c_void>(),
                log_decay_f32.cast::<c_void>(),
                plan.value_heads_u32,
                status.as_device_ptr().cast::<c_void>(),
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
    const TAIL_VALUE_HEADS: usize = 263;

    struct ScalarFixture {
        alpha: Vec<f32>,
        dt: Vec<f32>,
        a: Vec<f32>,
        beta_projection: Vec<f32>,
    }

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
        assert_deviates_beyond_tolerance(
            &beta,
            &wrong_reversed_sigmoid(&beta_projection),
            "reversed sigmoid",
        );
        assert_deviates_beyond_tolerance(
            &log_decay,
            &wrong_reexponentiated_decay(&alpha, &dt, &a),
            "re-exponentiated log decay",
        );
        assert_deviates_beyond_tolerance(
            &log_decay,
            &wrong_dt_omitted_decay(&alpha, &a),
            "dt-omitted log decay",
        );
        assert_deviates_beyond_tolerance(
            &log_decay,
            &wrong_dt_outside_softplus_decay(&alpha, &dt, &a),
            "dt-misplaced log decay",
        );

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
    fn cpu_only_reference_preserves_negative_overflow_softplus_saturation()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        // WHY: accepted CPU recurrence retains this saturation, while the raw native ABI excludes it.
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
    fn native_order_reference_preserves_tail_head_order()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let fixture = tail_fixture()?;
        let plan = RecurrentScalarsF32Plan::try_from_value_heads(TAIL_VALUE_HEADS)?;
        let (beta, log_decay) = recurrent_scalars_native_order_reference(
            plan,
            &fixture.alpha,
            &fixture.dt,
            &fixture.a,
            &fixture.beta_projection,
        )?;
        let (expected_beta, expected_log_decay) = recurrent_scalars_f64_oracle(
            &fixture.alpha,
            &fixture.dt,
            &fixture.a,
            &fixture.beta_projection,
        );
        assert_close_f64(&beta, &expected_beta, "tail recurrent beta");
        assert_close_f64(&log_decay, &expected_log_decay, "tail recurrent log decay");
        assert!(
            beta[255] < beta[256] && beta[256] < beta[257],
            "tail heads must retain their distinct beta-projection order"
        );
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

    #[test]
    #[ignore = "requires an operator-reserved gfx1100 device; source tests do not qualify hardware"]
    fn reserved_gfx1100_recurrent_scalars_match_independent_f64_oracle_with_tail()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer, Stream};

        const OUTPUT_SENTINEL: f32 = -1_234.5;

        let fixture = tail_fixture()?;
        let plan = RecurrentScalarsF32Plan::try_from_value_heads(TAIL_VALUE_HEADS)
            .map_err(|error| format!("plan recurrent scalars: {error}"))?;
        let (expected_beta, expected_log_decay) = recurrent_scalars_f64_oracle(
            &fixture.alpha,
            &fixture.dt,
            &fixture.a,
            &fixture.beta_projection,
        );
        let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;
        let alpha_device = DeviceBuffer::from_host(&device, &fixture.alpha)
            .map_err(|error| format!("upload alpha: {error}"))?;
        let dt_device = DeviceBuffer::from_host(&device, &fixture.dt)
            .map_err(|error| format!("upload dt: {error}"))?;
        let a_device = DeviceBuffer::from_host(&device, &fixture.a)
            .map_err(|error| format!("upload A: {error}"))?;
        let beta_projection_device = DeviceBuffer::from_host(&device, &fixture.beta_projection)
            .map_err(|error| format!("upload beta projection: {error}"))?;
        let sentinel = vec![OUTPUT_SENTINEL; TAIL_VALUE_HEADS];
        let beta_device = DeviceBuffer::from_host(&device, &sentinel)
            .map_err(|error| format!("initialize beta output: {error}"))?;
        let log_decay_device = DeviceBuffer::from_host(&device, &sentinel)
            .map_err(|error| format!("initialize log-decay output: {error}"))?;

        // SAFETY: all six exact spans are distinct owned device buffers and remain live through synchronization.
        unsafe {
            launch_recurrent_scalars_f32(
                plan,
                alpha_device.as_device_ptr(),
                alpha_device.len(),
                dt_device.as_device_ptr(),
                dt_device.len(),
                a_device.as_device_ptr(),
                a_device.len(),
                beta_projection_device.as_device_ptr(),
                beta_projection_device.len(),
                beta_device.as_device_ptr(),
                beta_device.len(),
                log_decay_device.as_device_ptr(),
                log_decay_device.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch recurrent scalars: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize recurrent scalars: {error}"))?;
        let mut beta = vec![0.0_f32; TAIL_VALUE_HEADS];
        beta_device
            .copy_to_host(&mut beta)
            .map_err(|error| format!("read beta output: {error}"))?;
        let mut log_decay = vec![0.0_f32; TAIL_VALUE_HEADS];
        log_decay_device
            .copy_to_host(&mut log_decay)
            .map_err(|error| format!("read log-decay output: {error}"))?;
        for (name, values) in [("beta", &beta), ("log decay", &log_decay)] {
            assert!(
                values.iter().all(|value| value.is_finite()),
                "{name} output must be finite"
            );
            assert!(
                values.iter().all(|value| *value != OUTPUT_SENTINEL),
                "{name} output must overwrite every nonzero sentinel"
            );
        }
        assert_close_f64(&beta, &expected_beta, "device recurrent beta");
        assert_close_f64(
            &log_decay,
            &expected_log_decay,
            "device recurrent log decay",
        );
        Ok(())
    }

    fn tail_fixture() -> core::result::Result<ScalarFixture, String> {
        let lane_values = (0..TAIL_VALUE_HEADS)
            .map(|index| {
                u16::try_from(index)
                    .map(f32::from)
                    .map_err(|error| format!("convert fixture lane {index}: {error}"))
            })
            .collect::<core::result::Result<Vec<_>, _>>()?;
        Ok(ScalarFixture {
            alpha: lane_values.iter().map(|lane| lane * 0.031 - 4.0).collect(),
            dt: lane_values
                .iter()
                .map(|lane| (lane % 11.0) * 0.07 - 0.35)
                .collect(),
            a: lane_values
                .iter()
                .map(|lane| (lane % 7.0) * 0.2 - 0.6)
                .collect(),
            beta_projection: lane_values.iter().map(|lane| lane * 0.043 - 5.5).collect(),
        })
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

    fn assert_deviates_beyond_tolerance(actual: &[f32], wrong: &[f64], operation: &str) {
        assert_eq!(actual.len(), wrong.len(), "{operation} output length");
        assert!(
            actual
                .iter()
                .zip(wrong.iter())
                .any(|(actual, wrong)| (f64::from(*actual) - wrong).abs() > f64::from(TOLERANCE)),
            "{operation} must diverge from the admitted recurrence beyond tolerance"
        );
    }

    fn wrong_reversed_sigmoid(beta_projection: &[f32]) -> Vec<f64> {
        beta_projection
            .iter()
            .map(|projection| {
                let projection = f64::from(*projection);
                1.0 / (1.0 + projection.exp())
            })
            .collect()
    }

    fn wrong_reexponentiated_decay(alpha: &[f32], dt: &[f32], a: &[f32]) -> Vec<f64> {
        alpha
            .iter()
            .zip(dt.iter())
            .zip(a.iter())
            .map(|((alpha, dt), a)| {
                let sum = f64::from(*alpha) + f64::from(*dt);
                let softplus = sum.max(0.0) + (-sum.abs()).exp().ln_1p();
                f64::from(*a) * softplus.exp()
            })
            .collect()
    }

    fn wrong_dt_omitted_decay(alpha: &[f32], a: &[f32]) -> Vec<f64> {
        alpha
            .iter()
            .zip(a.iter())
            .map(|(alpha, a)| {
                let alpha = f64::from(*alpha);
                let softplus = alpha.max(0.0) + (-alpha.abs()).exp().ln_1p();
                f64::from(*a) * softplus
            })
            .collect()
    }

    fn wrong_dt_outside_softplus_decay(alpha: &[f32], dt: &[f32], a: &[f32]) -> Vec<f64> {
        alpha
            .iter()
            .zip(dt.iter())
            .zip(a.iter())
            .map(|((alpha, dt), a)| {
                let alpha = f64::from(*alpha);
                let softplus = alpha.max(0.0) + (-alpha.abs()).exp().ln_1p();
                f64::from(*a) * softplus + f64::from(*dt)
            })
            .collect()
    }
}
