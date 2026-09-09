//! Checked private device spans shared by staged dense-f32 launchers.

use crate::error::{Result, UnsupportedShapeSnafu};

/// A nonempty checked device span.
#[derive(Clone, Copy)]
pub(crate) struct DeviceSpan {
    start: usize,
    end: usize,
    name: &'static str,
}

/// Check one raw dense-f32 device extent.
///
/// A zero-element extent has no footprint and therefore needs no pointer
/// representation. Nonempty extents must be non-null, aligned, and fit both
/// Rust's allocation-layout and address domains.
pub(crate) fn checked_f32_device_span(
    kernel: &'static str,
    pointer: *const f32,
    elements: usize,
    name: &'static str,
) -> Result<Option<DeviceSpan>> {
    if elements == 0 {
        return Ok(None);
    }
    if pointer.is_null() {
        return unsupported_shape(kernel, format!("{name} must be non-null"));
    }
    if !pointer.addr().is_multiple_of(core::mem::align_of::<f32>()) {
        return unsupported_shape(kernel, format!("{name} must be aligned for f32"));
    }
    let layout = std::alloc::Layout::array::<f32>(elements).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel,
            msg: format!("{name} length {elements} exceeds the Rust allocation layout domain"),
        }
        .build()
    })?;
    let start = pointer.addr();
    let end = start.checked_add(layout.size()).ok_or_else(|| {
        UnsupportedShapeSnafu {
            kernel,
            msg: format!("{name} device span overflows the address domain"),
        }
        .build()
    })?;
    Ok(Some(DeviceSpan { start, end, name }))
}

/// Refuse overlap involving a writable device span.
pub(crate) fn reject_overlapping_f32_spans(
    kernel: &'static str,
    left: Option<DeviceSpan>,
    right: Option<DeviceSpan>,
) -> Result<()> {
    let (Some(left), Some(right)) = (left, right) else {
        return Ok(());
    };
    if left.start < right.end && right.start < left.end {
        unsupported_shape(
            kernel,
            format!("writable {} span aliases {} span", left.name, right.name),
        )
    } else {
        Ok(())
    }
}

fn unsupported_shape<T>(kernel: &'static str, msg: String) -> Result<T> {
    UnsupportedShapeSnafu { kernel, msg }.fail()
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "device_span_test";

    #[test]
    fn empty_span_has_no_overlap_footprint()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let values = [0.0_f32; 2];
        let present = checked_f32_device_span(KERNEL, values.as_ptr(), values.len(), "present")?;
        let absent = checked_f32_device_span(
            KERNEL,
            values.as_ptr().wrapping_add(1),
            0,
            "absent inside present",
        )?;

        assert!(absent.is_none(), "zero elements must have no device footprint");
        reject_overlapping_f32_spans(KERNEL, absent, present)?;
        reject_overlapping_f32_spans(KERNEL, present, absent)?;
        Ok(())
    }
}
