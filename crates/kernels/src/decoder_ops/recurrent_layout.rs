//! Checked T=1 Qwen recurrent Q/K arrangement and L2 normalization.
//!
//! The operation consumes the already-SiLU-activated causal-convolution row.
//! It selects its leading Q and K source spans and writes the Qwen-specific
//! modulo-tiled, head-major `[value_heads, key_width]` outputs required by the
//! staged GDN step. It deliberately does not arrange V: for `T = 1`, V's
//! contiguous convolution tail already has the GDN row layout.

#[cfg(not(logismos_no_gpu_kernels))]
use std::ffi::c_void;

use hipcore::Stream;
#[cfg(test)]
use snafu::ResultExt;

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
use crate::device_span::{checked_f32_device_span, reject_overlapping_f32_spans};
#[cfg(not(logismos_no_gpu_kernels))]
use crate::error::LaunchSnafu;
#[cfg(logismos_no_gpu_kernels)]
use crate::error::NoGpuBuildSnafu;
use crate::error::{Result, UnsupportedShapeSnafu};

const RECURRENT_QK_L2_KERNEL: &str = "decoder_recurrent_qk_l2_f32";

#[cfg(not(logismos_no_gpu_kernels))]
unsafe extern "C" {
    fn logismos_launch_decoder_recurrent_qk_l2_f32(
        convolved_f32: *const c_void,
        query_f32: *mut c_void,
        key_f32: *mut c_void,
        source_key_heads: u32,
        value_heads: u32,
        key_width: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> u32;
}

/// Checked T=1 Q/K source and modulo-tiled output geometry for Qwen recurrence.
///
/// The two source spans are consecutive within an already-SiLU convolution
/// row: Q is `[0, source_elements)` and K immediately follows it. Each target
/// value head reads source head `value_head % source_key_heads`; this is Qwen's
/// pinned tiling order, which differs from grouped GDN's floor mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecurrentQkL2F32Plan {
    convolved_elements: usize,
    source_key_heads: usize,
    value_heads: usize,
    key_width: usize,
    source_elements: usize,
    output_elements: usize,
    epsilon: f32,
    source_key_heads_u32: u32,
    value_heads_u32: u32,
    key_width_u32: u32,
}

impl RecurrentQkL2F32Plan {
    /// Admit one already-activated T=1 convolution row and its Qwen Q/K layout.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when a dimension is zero,
    /// target heads are not divisible by source key heads, the two leading
    /// source spans do not fit the convolution row, a product or Rust f32
    /// layout overflows, a HIP ABI dimension is unrepresentable, or `epsilon`
    /// is not positive normal finite f32.
    pub fn try_from_dimensions(
        convolved_elements: usize,
        source_key_heads: usize,
        value_heads: usize,
        key_width: usize,
        epsilon: f32,
    ) -> Result<Self> {
        validate_nonzero("convolved_elements", convolved_elements)?;
        validate_nonzero("source_key_heads", source_key_heads)?;
        validate_nonzero("value_heads", value_heads)?;
        validate_nonzero("key_width", key_width)?;
        validate_positive_normal("epsilon", epsilon)?;
        if !value_heads.is_multiple_of(source_key_heads) {
            return unsupported_shape(format!(
                "value_heads {value_heads} must divide evenly by source_key_heads {source_key_heads}"
            ));
        }

        let source_elements =
            checked_product(source_key_heads, key_width, "source_key_heads * key_width")?;
        let qk_source_elements = checked_product(source_elements, 2, "Q/K source elements")?;
        if convolved_elements < qk_source_elements {
            return unsupported_shape(format!(
                "convolved row length {convolved_elements} cannot contain Q/K source prefix {qk_source_elements}"
            ));
        }
        let output_elements = checked_product(value_heads, key_width, "value_heads * key_width")?;
        validate_f32_layout("convolved row", convolved_elements)?;
        validate_f32_layout("Q source", source_elements)?;
        validate_f32_layout("K source", source_elements)?;
        validate_f32_layout("tiled Q output", output_elements)?;
        validate_f32_layout("tiled K output", output_elements)?;

        Ok(Self {
            convolved_elements,
            source_key_heads,
            value_heads,
            key_width,
            source_elements,
            output_elements,
            epsilon,
            source_key_heads_u32: abi_u32("source_key_heads", source_key_heads)?,
            value_heads_u32: abi_u32("value_heads", value_heads)?,
            key_width_u32: abi_u32("key_width", key_width)?,
        })
    }

    /// Return the exact already-activated convolution-row extent.
    #[must_use]
    pub const fn convolved_elements(self) -> usize {
        self.convolved_elements
    }

    /// Return the source Q/K head count before Qwen modulo tiling.
    #[must_use]
    pub const fn source_key_heads(self) -> usize {
        self.source_key_heads
    }

    /// Return the target value-head count after Qwen modulo tiling.
    #[must_use]
    pub const fn value_heads(self) -> usize {
        self.value_heads
    }

    /// Return the per-head Q/K width.
    #[must_use]
    pub const fn key_width(self) -> usize {
        self.key_width
    }

    /// Return the exact extent of either grouped Q or K source span.
    #[must_use]
    pub const fn source_elements(self) -> usize {
        self.source_elements
    }

    /// Return the exact extent of each tiled and normalized Q or K output.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output_elements
    }

    /// Return the L2 denominator lower bound.
    #[must_use]
    pub const fn epsilon(self) -> f32 {
        self.epsilon
    }
}

/// Launch T=1 Qwen Q/K selection, L2 normalization, and modulo tiling.
///
/// The source must be the already-SiLU-activated convolution row, not raw
/// causal-convolution output. The resulting Q/K rows are `[value_heads,
/// key_width]` in `value_head % source_key_heads` order; they intentionally
/// feed grouped GDN with equal key/value head counts.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] for a declared extent, pointer,
/// alignment, or writable-aliasing violation; [`crate::Error::NoGpuBuild`] in
/// a CPU-only build; and propagated stream-current or HIP launch failures.
///
/// # Safety
///
/// Every pointer must identify a live, correctly aligned allocation on
/// `stream`'s device for its exact declared plan extent through completion.
/// `convolved_f32` remains immutable and the two outputs remain exclusively
/// writable and mutually disjoint through completion. Inputs and every square,
/// lane sum, root, denominator, and normalized result must be finite and
/// normal-or-zero; this asynchronous ABI has no device status channel for CPU
/// numerical-domain refusals.
pub unsafe fn launch_recurrent_qk_l2_f32(
    plan: RecurrentQkL2F32Plan,
    convolved_f32: *const f32,
    convolved_elements: usize,
    query_f32: *mut f32,
    query_elements: usize,
    key_f32: *mut f32,
    key_elements: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            convolved_f32,
            convolved_elements,
            query_f32,
            query_elements,
            key_f32,
            key_elements,
            stream,
        );
        no_gpu_refusal()
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_launch(
            plan,
            convolved_f32,
            convolved_elements,
            query_f32,
            query_elements,
            key_f32,
            key_elements,
        )?;
        stream.make_current()?;
        // SAFETY: exact spans, non-aliasing writable outputs, and ABI-sized
        // geometry were checked locally; the caller upholds device ownership,
        // lifetime, and finite normal-or-zero arithmetic obligations.
        let code = unsafe {
            logismos_launch_decoder_recurrent_qk_l2_f32(
                convolved_f32.cast::<c_void>(),
                query_f32.cast::<c_void>(),
                key_f32.cast::<c_void>(),
                plan.source_key_heads_u32,
                plan.value_heads_u32,
                plan.key_width_u32,
                plan.epsilon,
                stream.raw().cast::<c_void>(),
            )
        };
        launch_result(code)
    }
}

#[cfg(logismos_no_gpu_kernels)]
fn no_gpu_refusal() -> Result<()> {
    NoGpuBuildSnafu {
        kernel: RECURRENT_QK_L2_KERNEL,
    }
    .fail()
}

#[cfg(not(logismos_no_gpu_kernels))]
fn launch_result(code: u32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        LaunchSnafu {
            kernel: RECURRENT_QK_L2_KERNEL,
            kind: hipcore::ErrorKind::from_raw(code),
            code,
        }
        .fail()
    }
}

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
fn validate_launch(
    plan: RecurrentQkL2F32Plan,
    convolved_f32: *const f32,
    convolved_elements: usize,
    query_f32: *mut f32,
    query_elements: usize,
    key_f32: *mut f32,
    key_elements: usize,
) -> Result<()> {
    validate_length("convolved row", convolved_elements, plan.convolved_elements)?;
    validate_length("tiled Q output", query_elements, plan.output_elements)?;
    validate_length("tiled K output", key_elements, plan.output_elements)?;
    let convolved = checked_f32_device_span(
        RECURRENT_QK_L2_KERNEL,
        convolved_f32,
        convolved_elements,
        "convolved row",
    )?;
    let query = checked_f32_device_span(
        RECURRENT_QK_L2_KERNEL,
        query_f32.cast_const(),
        query_elements,
        "tiled Q output",
    )?;
    let key = checked_f32_device_span(
        RECURRENT_QK_L2_KERNEL,
        key_f32.cast_const(),
        key_elements,
        "tiled K output",
    )?;
    reject_overlapping_f32_spans(RECURRENT_QK_L2_KERNEL, query, convolved)?;
    reject_overlapping_f32_spans(RECURRENT_QK_L2_KERNEL, key, convolved)?;
    reject_overlapping_f32_spans(RECURRENT_QK_L2_KERNEL, query, key)
}

fn validate_nonzero(name: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        unsupported_shape(format!("{name} must be greater than zero"))
    } else {
        Ok(())
    }
}

fn validate_positive_normal(name: &'static str, value: f32) -> Result<()> {
    if value.is_normal() && value.is_sign_positive() {
        Ok(())
    } else {
        unsupported_shape(format!("{name} must be positive normal finite f32"))
    }
}

fn checked_product(left: usize, right: usize, name: &'static str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        UnsupportedShapeSnafu {
            kernel: RECURRENT_QK_L2_KERNEL,
            msg: format!("{name} overflows usize"),
        }
        .build()
    })
}

fn validate_f32_layout(name: &'static str, elements: usize) -> Result<()> {
    std::alloc::Layout::array::<f32>(elements).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: RECURRENT_QK_L2_KERNEL,
            msg: format!("{name} length {elements} exceeds the Rust allocation layout domain"),
        }
        .build()
    })?;
    Ok(())
}

fn abi_u32(name: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: RECURRENT_QK_L2_KERNEL,
            msg: format!("{name} {value} exceeds the u32 HIP ABI"),
        }
        .build()
    })
}

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
fn validate_length(name: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        unsupported_shape(format!(
            "{name} length {actual} must equal checked {expected}"
        ))
    }
}

fn unsupported_shape<T>(msg: String) -> Result<T> {
    UnsupportedShapeSnafu {
        kernel: RECURRENT_QK_L2_KERNEL,
        msg,
    }
    .fail()
}

#[cfg(test)]
fn reserve_reference(elements: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .context(crate::error::CpuF32AllocationSnafu {
            operation: "recurrent Q/K L2 reference",
            requested_len: elements,
        })?;
    Ok(values)
}

#[cfg(test)]
fn native_order_reference(
    plan: RecurrentQkL2F32Plan,
    convolved: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    validate_length("convolved row", convolved.len(), plan.convolved_elements)?;
    let mut query = reserve_reference(plan.output_elements)?;
    let mut key = reserve_reference(plan.output_elements)?;
    for value_head in 0..plan.value_heads {
        let source_head = value_head % plan.source_key_heads;
        let source_start = checked_product(source_head, plan.key_width, "source head offset")?;
        let source_end = source_start.checked_add(plan.key_width).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: RECURRENT_QK_L2_KERNEL,
                msg: "source head end overflows usize".to_owned(),
            }
            .build()
        })?;
        let key_start = plan
            .source_elements
            .checked_add(source_start)
            .ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: RECURRENT_QK_L2_KERNEL,
                    msg: "K source head offset overflows usize".to_owned(),
                }
                .build()
            })?;
        let key_end = key_start.checked_add(plan.key_width).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: RECURRENT_QK_L2_KERNEL,
                msg: "K source head end overflows usize".to_owned(),
            }
            .build()
        })?;
        let query_source = convolved.get(source_start..source_end).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: RECURRENT_QK_L2_KERNEL,
                msg: "checked Q source slice access failed".to_owned(),
            }
            .build()
        })?;
        let key_source = convolved.get(key_start..key_end).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: RECURRENT_QK_L2_KERNEL,
                msg: "checked K source slice access failed".to_owned(),
            }
            .build()
        })?;
        let (query_denominator, key_denominator) =
            native_denominators(query_source, key_source, plan.epsilon);
        for value in query_source {
            query.push(*value / query_denominator);
        }
        for value in key_source {
            key.push(*value / key_denominator);
        }
    }
    Ok((query, key))
}

#[cfg(test)]
fn native_denominators(query: &[f32], key: &[f32], epsilon: f32) -> (f32, f32) {
    let query_sum = serial_sum_squares(query);
    let key_sum = serial_sum_squares(key);
    (query_sum.sqrt().max(epsilon), key_sum.sqrt().max(epsilon))
}

#[cfg(test)]
fn serial_sum_squares(values: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for value in values {
        sum += value * value;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-3;
    const F32_TOLERANCE: f32 = 1e-3;
    const EPSILON: f32 = 0.25;

    #[test]
    fn qwen_modulo_tiles_l2_normalized_heads_without_floor_grouping() -> Result<()> {
        let plan = RecurrentQkL2F32Plan::try_from_dimensions(16, 2, 4, 3, 1e-5)?;
        let convolved = [
            3.0_f32, 4.0, 12.0, 5.0, 12.0, 0.0, -8.0, 6.0, 0.0, 9.0, 12.0, 20.0, 71.0, 72.0, 73.0,
            74.0,
        ];
        let (query, key) = native_order_reference(plan, &convolved)?;
        let (expected_query, expected_key) = f64_logical_oracle(plan, &convolved)?;
        assert_close_f64(&query, &expected_query, "Q native order");
        assert_close_f64(&key, &expected_key, "K native order");

        let width = plan.key_width();
        let q0 = query
            .get(..width)
            .ok_or_else(|| missing_test_slice("Q head zero"))?;
        let q1 = query
            .get(width..width * 2)
            .ok_or_else(|| missing_test_slice("Q head one"))?;
        let q2 = query
            .get(width * 2..width * 3)
            .ok_or_else(|| missing_test_slice("Q head two"))?;
        let q3 = query
            .get(width * 3..width * 4)
            .ok_or_else(|| missing_test_slice("Q head three"))?;
        assert_eq!(q0, q2, "head two must repeat source head zero");
        assert_eq!(q1, q3, "head three must repeat source head one");
        let separation = q0
            .first()
            .zip(q1.first())
            .map(|(left, right)| (left - right).abs())
            .ok_or_else(|| missing_test_slice("Q head markers"))?;
        assert!(
            separation > F32_TOLERANCE,
            "source heads must distinguish modulo [0, 1, 0, 1] from floor [0, 0, 1, 1]"
        );
        Ok(())
    }

    #[test]
    fn l2_denominator_applies_epsilon_after_the_unweighted_root() -> Result<()> {
        let plan = RecurrentQkL2F32Plan::try_from_dimensions(5, 1, 1, 2, EPSILON)?;
        let convolved = [0.125_f32, 0.0, 0.0625, 0.0, 99.0];
        let (query, key) = native_order_reference(plan, &convolved)?;
        let expected_query = 0.5_f32;
        let expected_key = 0.25_f32;
        let actual_query = query
            .first()
            .copied()
            .ok_or_else(|| missing_test_slice("Q output"))?;
        let actual_key = key
            .first()
            .copied()
            .ok_or_else(|| missing_test_slice("K output"))?;
        assert!((actual_query - expected_query).abs() <= f32::EPSILON);
        assert!((actual_key - expected_key).abs() <= f32::EPSILON);
        let wrong_inside_root = 0.125_f32 / (0.125_f32 * 0.125_f32 + EPSILON).sqrt();
        assert!(
            (actual_query - wrong_inside_root).abs() > F32_TOLERANCE,
            "epsilon must lower-bound sqrt(sum of squares), not enter its radicand"
        );
        Ok(())
    }

    #[test]
    fn native_order_reference_broadcasts_serial_denominators_to_every_output_column() -> Result<()>
    {
        let plan = RecurrentQkL2F32Plan::try_from_dimensions(77, 1, 1, 37, 1e-5)?;
        let mut convolved = vec![0.0_f32; plan.convolved_elements()];
        let values = [1.0_f32, 1.25, 1.5, 1.75];
        for column in 0..plan.key_width() {
            let value = values
                .get(column % values.len())
                .copied()
                .ok_or_else(|| missing_test_slice("nonzero tail value"))?;
            let query_slot = convolved
                .get_mut(column)
                .ok_or_else(|| missing_test_slice("Q output-tail slot"))?;
            *query_slot = value;
            let key_column = plan
                .source_elements()
                .checked_add(column)
                .ok_or_else(|| missing_test_slice("K output-tail column"))?;
            let key_slot = convolved
                .get_mut(key_column)
                .ok_or_else(|| missing_test_slice("K output-tail slot"))?;
            *key_slot = value + 1.0;
        }
        let (query, key) = native_order_reference(plan, &convolved)?;
        let (expected_query, expected_key) = f64_logical_oracle(plan, &convolved)?;
        assert_close_f64(&query, &expected_query, "Q output-tail native order");
        assert_close_f64(&key, &expected_key, "K output-tail native order");
        let query_source = convolved
            .get(..plan.source_elements())
            .ok_or_else(|| missing_test_slice("Q serial source"))?;
        let key_source = convolved
            .get(plan.source_elements()..plan.source_elements() * 2)
            .ok_or_else(|| missing_test_slice("K serial source"))?;
        let query_denominator = serial_sum_squares(query_source).sqrt().max(plan.epsilon());
        let key_denominator = serial_sum_squares(key_source).sqrt().max(plan.epsilon());
        for column in 0..plan.key_width() {
            let query_input = query_source
                .get(column)
                .copied()
                .ok_or_else(|| missing_test_slice("Q broadcast input"))?;
            let key_input = key_source
                .get(column)
                .copied()
                .ok_or_else(|| missing_test_slice("K broadcast input"))?;
            let query_output = query
                .get(column)
                .copied()
                .ok_or_else(|| missing_test_slice("Q broadcast output"))?;
            let key_output = key
                .get(column)
                .copied()
                .ok_or_else(|| missing_test_slice("K broadcast output"))?;
            assert!(
                (query_output - query_input / query_denominator).abs() <= f32::EPSILON,
                "Q column {column} must use lane-zero's serial denominator"
            );
            assert!(
                (key_output - key_input / key_denominator).abs() <= f32::EPSILON,
                "K column {column} must use lane-zero's serial denominator"
            );
        }
        let tail_column = plan.key_width() - 1;
        let tail_input = query_source
            .get(tail_column)
            .copied()
            .ok_or_else(|| missing_test_slice("Q tail input"))?;
        let tail_output = query
            .get(tail_column)
            .copied()
            .ok_or_else(|| missing_test_slice("Q tail output"))?;
        let old_lane_local = tail_input / tail_input.abs();
        assert!(
            (tail_output - old_lane_local).abs() > F32_TOLERANCE,
            "a tail lane must not use its local suffix denominator"
        );
        assert!(
            RecurrentQkL2F32Plan::try_from_dimensions(4, 0, 1, 1, 1e-5).is_err(),
            "zero source head count must be rejected"
        );
        assert!(
            RecurrentQkL2F32Plan::try_from_dimensions(4, 2, 3, 1, 1e-5).is_err(),
            "nondivisible target/source grouping must be rejected"
        );
        assert!(
            RecurrentQkL2F32Plan::try_from_dimensions(3, 1, 1, 2, 1e-5).is_err(),
            "a row shorter than two Q/K source spans must be rejected"
        );
        assert!(
            RecurrentQkL2F32Plan::try_from_dimensions(4, 1, 1, 2, 0.0).is_err(),
            "nonpositive epsilon must be rejected"
        );
        assert!(
            RecurrentQkL2F32Plan::try_from_dimensions(usize::MAX, 1, 1, 2, 1e-5).is_err(),
            "unrepresentable f32 layouts must be rejected"
        );
        Ok(())
    }

    #[test]
    fn serial_f32_sum_preserves_the_lane_zero_accumulation_order() {
        let values = [4_096.0_f32, 1.0, 1.0];
        let serial = serial_sum_squares(&values);
        let regrouped = values[1] * values[1] + values[2] * values[2] + values[0] * values[0];
        assert_eq!(serial.to_bits(), 16_777_216.0_f32.to_bits());
        assert_eq!(regrouped.to_bits(), 16_777_218.0_f32.to_bits());
        assert_ne!(
            serial.to_bits(),
            regrouped.to_bits(),
            "the native baseline deliberately retains serial f32 accumulation"
        );
    }

    #[test]
    fn recurrent_qk_l2_validator_accepts_exact_spans_and_refuses_unsafe_bindings()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = RecurrentQkL2F32Plan::try_from_dimensions(16, 2, 4, 3, 1e-5)?;
        let convolved = [1.0_f32; 16];
        let mut query = [0.0_f32; 12];
        let mut key = [0.0_f32; 12];
        validate_launch(
            plan,
            convolved.as_ptr(),
            convolved.len(),
            query.as_mut_ptr(),
            query.len(),
            key.as_mut_ptr(),
            key.len(),
        )?;
        assert!(
            validate_launch(
                plan,
                convolved.as_ptr(),
                convolved.len() - 1,
                query.as_mut_ptr(),
                query.len(),
                key.as_mut_ptr(),
                key.len(),
            )
            .is_err(),
            "short convolved source span must be refused"
        );
        assert!(
            validate_launch(
                plan,
                convolved.as_ptr(),
                convolved.len(),
                query.as_mut_ptr(),
                query.len() - 1,
                key.as_mut_ptr(),
                key.len(),
            )
            .is_err(),
            "short Q output span must be refused"
        );
        assert!(
            validate_launch(
                plan,
                convolved.as_ptr(),
                convolved.len(),
                query.as_mut_ptr(),
                query.len(),
                key.as_mut_ptr(),
                key.len() - 1,
            )
            .is_err(),
            "short K output span must be refused"
        );
        assert!(
            validate_launch(
                plan,
                core::ptr::null(),
                convolved.len(),
                query.as_mut_ptr(),
                query.len(),
                key.as_mut_ptr(),
                key.len(),
            )
            .is_err(),
            "null convolved source must be refused"
        );
        assert!(
            validate_launch(
                plan,
                convolved.as_ptr(),
                convolved.len(),
                query.as_mut_ptr().wrapping_byte_add(1),
                query.len(),
                key.as_mut_ptr(),
                key.len(),
            )
            .is_err(),
            "misaligned Q output must be refused"
        );
        assert!(
            validate_launch(
                plan,
                convolved.as_ptr(),
                convolved.len(),
                convolved.as_ptr().cast_mut(),
                query.len(),
                key.as_mut_ptr(),
                key.len(),
            )
            .is_err(),
            "Q output must not alias the convolved source"
        );
        assert!(
            validate_launch(
                plan,
                convolved.as_ptr(),
                convolved.len(),
                query.as_mut_ptr(),
                query.len(),
                convolved.as_ptr().cast_mut(),
                key.len(),
            )
            .is_err(),
            "K output must not alias the convolved source"
        );
        assert!(
            validate_launch(
                plan,
                convolved.as_ptr(),
                convolved.len(),
                query.as_mut_ptr(),
                query.len(),
                query.as_mut_ptr(),
                key.len(),
            )
            .is_err(),
            "Q and K outputs must not alias each other"
        );
        Ok(())
    }

    #[cfg(logismos_no_gpu_kernels)]
    #[test]
    fn cpu_only_build_refuses_native_recurrent_launch() {
        assert!(
            matches!(no_gpu_refusal(), Err(crate::Error::NoGpuBuild { .. })),
            "native recurrent layout must retain the typed CPU-only refusal"
        );
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    #[test]
    #[ignore = "requires an operator-reserved HIP device; source tests do not qualify hardware"]
    fn reserved_device_recurrent_qk_l2_matches_native_and_independent_references()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer};

        let plan = RecurrentQkL2F32Plan::try_from_dimensions(16, 2, 4, 3, 1e-5)
            .map_err(|error| format!("plan recurrent Q/K L2: {error}"))?;
        let convolved_host = [
            3.0_f32, 4.0, 12.0, 5.0, 12.0, 0.0, -8.0, 6.0, 0.0, 9.0, 12.0, 20.0, 71.0, 72.0, 73.0,
            74.0,
        ];
        let output_sentinel = -1_234.5_f32;
        let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;
        let convolved = DeviceBuffer::from_host(&device, &convolved_host)
            .map_err(|error| format!("upload convolved row: {error}"))?;
        let output_host = vec![output_sentinel; plan.output_elements()];
        let query = DeviceBuffer::from_host(&device, &output_host)
            .map_err(|error| format!("initialize Q output: {error}"))?;
        let key = DeviceBuffer::from_host(&device, &output_host)
            .map_err(|error| format!("initialize K output: {error}"))?;
        // SAFETY: the distinct owned buffers have the exact plan extents and
        // remain live through this stream's synchronization.
        unsafe {
            launch_recurrent_qk_l2_f32(
                plan,
                convolved.as_device_ptr(),
                convolved.len(),
                query.as_device_ptr(),
                query.len(),
                key.as_device_ptr(),
                key.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch recurrent Q/K L2: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize recurrent Q/K L2: {error}"))?;
        let query_actual = read_device(&query)?;
        let key_actual = read_device(&key)?;
        let (query_native, key_native) = native_order_reference(plan, &convolved_host)
            .map_err(|error| format!("native Q/K L2 reference: {error}"))?;
        let (query_f64, key_f64) = f64_logical_oracle(plan, &convolved_host)
            .map_err(|error| format!("independent Q/K L2 oracle: {error}"))?;
        assert_close_f32(&query_actual, &query_native, "device Q native order");
        assert_close_f32(&key_actual, &key_native, "device K native order");
        assert_close_f64(&query_actual, &query_f64, "device Q independent f64");
        assert_close_f64(&key_actual, &key_f64, "device K independent f64");
        Ok(())
    }

    fn f64_logical_oracle(
        plan: RecurrentQkL2F32Plan,
        convolved: &[f32],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        validate_length("convolved row", convolved.len(), plan.convolved_elements())?;
        let mut query = Vec::with_capacity(plan.output_elements());
        let mut key = Vec::with_capacity(plan.output_elements());
        for value_head in 0..plan.value_heads() {
            let source_head = value_head % plan.source_key_heads();
            let source_start = source_head
                .checked_mul(plan.key_width())
                .ok_or_else(|| missing_test_slice("f64 source offset"))?;
            let source_end = source_start
                .checked_add(plan.key_width())
                .ok_or_else(|| missing_test_slice("f64 source end"))?;
            let key_start = plan
                .source_elements()
                .checked_add(source_start)
                .ok_or_else(|| missing_test_slice("f64 key offset"))?;
            let key_end = key_start
                .checked_add(plan.key_width())
                .ok_or_else(|| missing_test_slice("f64 key end"))?;
            let query_source = convolved
                .get(source_start..source_end)
                .ok_or_else(|| missing_test_slice("f64 Q source"))?;
            let key_source = convolved
                .get(key_start..key_end)
                .ok_or_else(|| missing_test_slice("f64 K source"))?;
            let query_sum = query_source
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>();
            let key_sum = key_source
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>();
            let query_denominator = query_sum.sqrt().max(f64::from(plan.epsilon()));
            let key_denominator = key_sum.sqrt().max(f64::from(plan.epsilon()));
            query.extend(
                query_source
                    .iter()
                    .map(|value| f64::from(*value) / query_denominator),
            );
            key.extend(
                key_source
                    .iter()
                    .map(|value| f64::from(*value) / key_denominator),
            );
        }
        Ok((query, key))
    }

    fn assert_close_f64(actual: &[f32], expected: &[f64], operation: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{operation} output lengths must match"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f64::from(*actual) - expected).abs() <= TOLERANCE,
                "{operation} index {index}: got {actual}, expected {expected}"
            );
        }
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    fn assert_close_f32(actual: &[f32], expected: &[f32], operation: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{operation} output lengths must match"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= F32_TOLERANCE,
                "{operation} index {index}: got {actual}, expected {expected}"
            );
        }
    }

    fn missing_test_slice(context: &'static str) -> crate::Error {
        UnsupportedShapeSnafu {
            kernel: RECURRENT_QK_L2_KERNEL,
            msg: format!("test fixture missing {context}"),
        }
        .build()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    fn read_device(buffer: &hipcore::DeviceBuffer<f32>) -> core::result::Result<Vec<f32>, String> {
        let mut host = vec![0.0_f32; buffer.len()];
        buffer
            .copy_to_host(&mut host)
            .map_err(|error| format!("copy device buffer: {error}"))?;
        Ok(host)
    }
}
