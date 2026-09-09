//! Checked device-side copies for a backend-owned paged K/V allocation.
//!
//! This module has no cache dependency: callers retain page-table authority and
//! provide the exact dense layer-major backing described by [`PagedKvNativeLayout`].

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
use hipcore::Stream;

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use crate::device_span::{
    checked_device_span, checked_f32_device_span, reject_overlapping_device_spans,
};
#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use crate::error::LaunchSnafu;
#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
use crate::error::NoGpuBuildSnafu;
use crate::error::{Result, UnsupportedShapeSnafu};

const KERNEL: &str = "paged_kv";

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe extern "C" {
    fn logismos_launch_paged_kv_copy_tail_f32(
        keys_f32: *mut c_void,
        values_f32: *mut c_void,
        layers: u32,
        row_width: u32,
        page_tokens: u32,
        physical_pages: u32,
        source_page: u32,
        destination_page: u32,
        filled_tokens: u32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_paged_kv_append_row_f32(
        keys_f32: *mut c_void,
        values_f32: *mut c_void,
        source_key_f32: *const c_void,
        source_value_f32: *const c_void,
        layer: u32,
        physical_page: u32,
        in_page_token: u32,
        row_width: u32,
        page_tokens: u32,
        physical_pages: u32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_paged_kv_write_table_u32(
        table_u32: *mut c_void,
        logical_page: u32,
        physical_page: u32,
        stream: *mut c_void,
    ) -> u32;
}

/// Checked layer-major native K/V backing geometry.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedKvNativeLayout {
    layers: usize,
    row_width: usize,
    page_tokens: usize,
    physical_pages: usize,
    page_elements: usize,
    layer_elements: usize,
    backing_elements: usize,
}

#[cfg(feature = "gpu")]
impl PagedKvNativeLayout {
    /// Admit a separate K or V allocation laid out as
    /// `[layer][physical_page][in_page_token][row_width]`.
    pub fn try_from_dimensions(
        layers: usize,
        row_width: usize,
        page_tokens: usize,
        physical_pages: usize,
    ) -> Result<Self> {
        for (name, value) in [
            ("layers", layers),
            ("row_width", row_width),
            ("page_tokens", page_tokens),
            ("physical_pages", physical_pages),
        ] {
            if value == 0 {
                return unsupported(format!("{name} must be nonzero"));
            }
            u32::try_from(value).map_err(|_| abi(name, value))?;
        }
        if !matches!(page_tokens, 8 | 16 | 32) {
            return unsupported(format!("page_tokens {page_tokens} must be B8, B16, or B32"));
        }
        let page_elements = checked_product(page_tokens, row_width, "page token row extent")?;
        let layer_elements = checked_product(physical_pages, page_elements, "layer page extent")?;
        let backing_elements = checked_product(layers, layer_elements, "all-layer backing extent")?;
        layout("one K/V backing", backing_elements)?;
        Ok(Self {
            layers,
            row_width,
            page_tokens,
            physical_pages,
            page_elements,
            layer_elements,
            backing_elements,
        })
    }

    /// Number of full-attention layers represented by the backing.
    #[must_use]
    pub const fn layers(self) -> usize {
        self.layers
    }

    /// Elements in a K/V row.
    #[must_use]
    pub const fn row_width(self) -> usize {
        self.row_width
    }

    /// Logical tokens in each physical native page.
    #[must_use]
    pub const fn page_tokens(self) -> usize {
        self.page_tokens
    }

    /// All-layer physical page bundles, including the COW spare.
    #[must_use]
    pub const fn physical_pages(self) -> usize {
        self.physical_pages
    }

    /// Elements in one physical page for one layer and K/V kind.
    #[must_use]
    pub const fn page_elements(self) -> usize {
        self.page_elements
    }

    /// Elements in all physical pages for one layer and K/V kind.
    #[must_use]
    pub const fn layer_elements(self) -> usize {
        self.layer_elements
    }

    /// Elements in one separate all-layer K or V backing.
    #[must_use]
    pub const fn backing_elements(self) -> usize {
        self.backing_elements
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    fn abi(self) -> Result<PagedKvNativeAbi> {
        Ok(PagedKvNativeAbi {
            layers: u32::try_from(self.layers).map_err(|_| abi("layers", self.layers))?,
            row_width: u32::try_from(self.row_width)
                .map_err(|_| abi("row_width", self.row_width))?,
            page_tokens: u32::try_from(self.page_tokens)
                .map_err(|_| abi("page_tokens", self.page_tokens))?,
            physical_pages: u32::try_from(self.physical_pages)
                .map_err(|_| abi("physical_pages", self.physical_pages))?,
        })
    }
}

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
#[derive(Clone, Copy)]
struct PagedKvNativeAbi {
    layers: u32,
    row_width: u32,
    page_tokens: u32,
    physical_pages: u32,
}

/// Copy the written prefix of one physical page for every layer and K/V kind.
///
/// # Safety
///
/// `keys_f32` and `values_f32` must be distinct live device allocations with
/// exactly [`PagedKvNativeLayout::backing_elements`] elements. The caller owns
/// stream completion and must not free or reuse either allocation until it is
/// known complete.
#[cfg(feature = "gpu")]
pub unsafe fn copy_tail_f32(
    layout: PagedKvNativeLayout,
    keys_f32: *mut f32,
    key_elements: usize,
    values_f32: *mut f32,
    value_elements: usize,
    source_page: usize,
    destination_page: usize,
    filled_tokens: usize,
    stream: &Stream,
) -> Result<()> {
    validate_copy(
        layout,
        key_elements,
        value_elements,
        source_page,
        destination_page,
        filled_tokens,
    )?;
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (keys_f32, values_f32, stream);
        NoGpuBuildSnafu { kernel: KERNEL }.fail()
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let keys = checked_f32_device_span(KERNEL, keys_f32, key_elements, "keys")?;
        let values = checked_f32_device_span(KERNEL, values_f32, value_elements, "values")?;
        reject_overlapping_device_spans(KERNEL, keys, values)?;
        let native_abi = layout.abi()?;
        stream.make_current()?;
        // SAFETY: exact spans and ABI dimensions were checked; caller retains device ownership through completion.
        let code = unsafe {
            logismos_launch_paged_kv_copy_tail_f32(
                keys_f32.cast::<c_void>(),
                values_f32.cast::<c_void>(),
                native_abi.layers,
                native_abi.row_width,
                native_abi.page_tokens,
                native_abi.physical_pages,
                u32::try_from(source_page).map_err(|_| abi("source_page", source_page))?,
                u32::try_from(destination_page)
                    .map_err(|_| abi("destination_page", destination_page))?,
                u32::try_from(filled_tokens).map_err(|_| abi("filled_tokens", filled_tokens))?,
                stream.raw().cast::<c_void>(),
            )
        };
        launch(code, "paged_kv_copy_tail_f32")
    }
}

/// Copy one device-resident K/V row into an unpublished native page location.
///
/// # Safety
///
/// Every pointer must identify a live, non-overlapping device span on
/// `stream`'s device. The caller owns completion and publication.
#[cfg(feature = "gpu")]
pub unsafe fn append_row_f32(
    layout: PagedKvNativeLayout,
    keys_f32: *mut f32,
    key_elements: usize,
    values_f32: *mut f32,
    value_elements: usize,
    source_key_f32: *const f32,
    source_key_elements: usize,
    source_value_f32: *const f32,
    source_value_elements: usize,
    layer: usize,
    physical_page: usize,
    in_page_token: usize,
    stream: &Stream,
) -> Result<()> {
    validate_append(
        layout,
        key_elements,
        value_elements,
        source_key_elements,
        source_value_elements,
        layer,
        physical_page,
        in_page_token,
    )?;
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            keys_f32,
            values_f32,
            source_key_f32,
            source_value_f32,
            stream,
        );
        NoGpuBuildSnafu { kernel: KERNEL }.fail()
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let keys = checked_f32_device_span(KERNEL, keys_f32, key_elements, "keys")?;
        let values = checked_f32_device_span(KERNEL, values_f32, value_elements, "values")?;
        let source_key =
            checked_f32_device_span(KERNEL, source_key_f32, source_key_elements, "source key")?;
        let source_value = checked_f32_device_span(
            KERNEL,
            source_value_f32,
            source_value_elements,
            "source value",
        )?;
        for input in [values, source_key, source_value] {
            reject_overlapping_device_spans(KERNEL, keys, input)?;
        }
        reject_overlapping_device_spans(KERNEL, values, source_key)?;
        reject_overlapping_device_spans(KERNEL, values, source_value)?;
        reject_overlapping_device_spans(KERNEL, source_key, source_value)?;
        let native_abi = layout.abi()?;
        stream.make_current()?;
        // SAFETY: validated spans and dimensions match the fixed device copy ABI.
        let code = unsafe {
            logismos_launch_paged_kv_append_row_f32(
                keys_f32.cast::<c_void>(),
                values_f32.cast::<c_void>(),
                source_key_f32.cast::<c_void>(),
                source_value_f32.cast::<c_void>(),
                u32::try_from(layer).map_err(|_| abi("layer", layer))?,
                u32::try_from(physical_page).map_err(|_| abi("physical_page", physical_page))?,
                u32::try_from(in_page_token).map_err(|_| abi("in_page_token", in_page_token))?,
                native_abi.row_width,
                native_abi.page_tokens,
                native_abi.physical_pages,
                stream.raw().cast::<c_void>(),
            )
        };
        launch(code, "paged_kv_append_row_f32")
    }
}

/// Mirror one staged logical-page mapping into a device `u32` table.
///
/// # Safety
///
/// `table_u32` must be a live device allocation with `table_entries` entries;
/// the caller retains it through stream completion.
#[cfg(feature = "gpu")]
pub unsafe fn write_table_u32(
    table_u32: *mut u32,
    table_entries: usize,
    logical_page: usize,
    physical_page: usize,
    stream: &Stream,
) -> Result<()> {
    if logical_page >= table_entries {
        return unsupported(format!(
            "logical page {logical_page} is outside table entries {table_entries}"
        ));
    }
    u32::try_from(logical_page).map_err(|_| abi("logical_page", logical_page))?;
    u32::try_from(physical_page).map_err(|_| abi("physical_page", physical_page))?;
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (table_u32, stream);
        NoGpuBuildSnafu { kernel: KERNEL }.fail()
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let _ = checked_device_span(KERNEL, table_u32, table_entries, "page table")?;
        stream.make_current()?;
        // SAFETY: device table span and ABI-sized indices were checked.
        let code = unsafe {
            logismos_launch_paged_kv_write_table_u32(
                table_u32.cast::<c_void>(),
                u32::try_from(logical_page).map_err(|_| abi("logical_page", logical_page))?,
                u32::try_from(physical_page).map_err(|_| abi("physical_page", physical_page))?,
                stream.raw().cast::<c_void>(),
            )
        };
        launch(code, "paged_kv_write_table_u32")
    }
}

#[cfg(feature = "gpu")]
fn validate_copy(
    layout: PagedKvNativeLayout,
    key_elements: usize,
    value_elements: usize,
    source_page: usize,
    destination_page: usize,
    filled_tokens: usize,
) -> Result<()> {
    validate_backing(layout, key_elements, value_elements)?;
    if source_page >= layout.physical_pages
        || destination_page >= layout.physical_pages
        || source_page == destination_page
    {
        return unsupported(
            "source and destination must be distinct admitted physical pages".to_string(),
        );
    }
    if filled_tokens == 0 || filled_tokens > layout.page_tokens {
        return unsupported(format!(
            "filled token count {filled_tokens} is outside one physical page"
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn validate_append(
    layout: PagedKvNativeLayout,
    key_elements: usize,
    value_elements: usize,
    source_key_elements: usize,
    source_value_elements: usize,
    layer: usize,
    physical_page: usize,
    in_page_token: usize,
) -> Result<()> {
    validate_backing(layout, key_elements, value_elements)?;
    if source_key_elements != layout.row_width || source_value_elements != layout.row_width {
        return unsupported(format!(
            "source K/V rows must both have width {}",
            layout.row_width
        ));
    }
    if layer >= layout.layers
        || physical_page >= layout.physical_pages
        || in_page_token >= layout.page_tokens
    {
        return unsupported(
            "append destination is outside admitted layer-major backing".to_string(),
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn validate_backing(
    layout: PagedKvNativeLayout,
    key_elements: usize,
    value_elements: usize,
) -> Result<()> {
    if key_elements != layout.backing_elements || value_elements != layout.backing_elements {
        return unsupported(format!(
            "K/V backing lengths must both equal {}",
            layout.backing_elements
        ));
    }
    Ok(())
}

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
fn launch(code: u32, kernel: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        LaunchSnafu {
            kernel,
            kind: hipcore::ErrorKind::from_raw(code),
            code,
        }
        .fail()
    }
}

#[cfg(feature = "gpu")]
fn checked_product(left: usize, right: usize, context: &'static str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        UnsupportedShapeSnafu {
            kernel: KERNEL,
            msg: format!("{context} overflows usize"),
        }
        .build()
    })
}

#[cfg(feature = "gpu")]
fn layout(name: &'static str, elements: usize) -> Result<()> {
    std::alloc::Layout::array::<f32>(elements).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: KERNEL,
            msg: format!("{name} layout for {elements} f32 elements is unrepresentable"),
        }
        .build()
    })?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn abi(name: &'static str, value: usize) -> crate::Error {
    UnsupportedShapeSnafu {
        kernel: KERNEL,
        msg: format!("{name} {value} exceeds the u32 HIP ABI"),
    }
    .build()
}

#[cfg(feature = "gpu")]
fn unsupported<T>(msg: String) -> Result<T> {
    UnsupportedShapeSnafu {
        kernel: KERNEL,
        msg,
    }
    .fail()
}

#[cfg(test)]
mod tests {
    use super::PagedKvNativeLayout;

    fn reference_offset(
        layer: usize,
        page: usize,
        token: usize,
        column: usize,
        layout: PagedKvNativeLayout,
    ) -> usize {
        (((layer * layout.physical_pages() + page) * layout.page_tokens() + token)
            * layout.row_width())
            + column
    }

    fn reference_copy_tail(
        keys: &mut [f32],
        values: &mut [f32],
        layout: PagedKvNativeLayout,
        source_page: usize,
        destination_page: usize,
        filled_tokens: usize,
    ) {
        for layer in 0..layout.layers() {
            for token in 0..filled_tokens {
                for column in 0..layout.row_width() {
                    let source = reference_offset(layer, source_page, token, column, layout);
                    let destination =
                        reference_offset(layer, destination_page, token, column, layout);
                    keys[destination] = keys[source];
                    values[destination] = values[source];
                }
            }
        }
    }

    fn reference_append_row(
        keys: &mut [f32],
        values: &mut [f32],
        key: &[f32],
        value: &[f32],
        layout: PagedKvNativeLayout,
        layer: usize,
        page: usize,
        token: usize,
    ) {
        for column in 0..layout.row_width() {
            let destination = reference_offset(layer, page, token, column, layout);
            keys[destination] = key[column];
            values[destination] = value[column];
        }
    }

    fn reference_write_table(table: &mut [u32], logical_page: usize, physical_page: u32) {
        table[logical_page] = physical_page;
    }

    #[test]
    fn independent_reference_observes_layer_major_copy_append_and_table_domains()
    -> Result<(), Box<dyn std::error::Error>> {
        let layout = PagedKvNativeLayout::try_from_dimensions(2, 3, 8, 4)?;
        let mut keys = (0..layout.backing_elements())
            .map(|index| index as f32 + 1.0)
            .collect::<Vec<_>>();
        let mut values = (0..layout.backing_elements())
            .map(|index| -(index as f32) - 1.0)
            .collect::<Vec<_>>();
        let key = [101.0_f32, 102.0, 103.0];
        let value = [-101.0_f32, -102.0, -103.0];
        reference_append_row(&mut keys, &mut values, &key, &value, layout, 1, 2, 5);
        let row_start = reference_offset(1, 2, 5, 0, layout);
        assert_eq!(&keys[row_start..row_start + key.len()], key);
        assert_eq!(&values[row_start..row_start + value.len()], value);

        let before_tail_keys = keys.clone();
        let before_tail_values = values.clone();
        reference_copy_tail(&mut keys, &mut values, layout, 2, 3, 6);
        for layer in 0..layout.layers() {
            for token in 0..6 {
                for column in 0..layout.row_width() {
                    let source = reference_offset(layer, 2, token, column, layout);
                    let destination = reference_offset(layer, 3, token, column, layout);
                    assert_eq!(keys[destination], before_tail_keys[source]);
                    assert_eq!(values[destination], before_tail_values[source]);
                }
            }
        }
        let untouched = reference_offset(1, 3, 6, 0, layout);
        assert_eq!(keys[untouched], before_tail_keys[untouched]);
        assert_eq!(values[untouched], before_tail_values[untouched]);
        let mut table = vec![0_u32; 3];
        reference_write_table(&mut table, 1, 3);
        assert_eq!(table, [0, 3, 0]);
        Ok(())
    }

    #[test]
    fn layout_refuses_noncanonical_page_and_overflowing_backing()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(PagedKvNativeLayout::try_from_dimensions(1, 1, 12, 1).is_err());
        assert!(PagedKvNativeLayout::try_from_dimensions(usize::MAX, 1, 8, 1).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
    fn reserved_device_append_and_cow_match_independent_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        use hipcore::{Device, DeviceBuffer, Stream};

        let layout = PagedKvNativeLayout::try_from_dimensions(1, 3, 8, 3)?;
        let device = Device::new(0)?;
        let stream = Stream::new(&device)?;
        let initial_keys = (0..layout.backing_elements())
            .map(|index| index as f32 + 1.0)
            .collect::<Vec<_>>();
        let initial_values = (0..layout.backing_elements())
            .map(|index| -(index as f32) - 1.0)
            .collect::<Vec<_>>();
        let key = [51.0_f32, 52.0, 53.0];
        let value = [-51.0_f32, -52.0, -53.0];
        let expected_key = key.to_vec();
        let expected_value = value.to_vec();
        let mut reference_table = vec![0_u32; 2];
        let mut reference_keys = initial_keys.clone();
        let mut reference_values = initial_values.clone();
        reference_append_row(
            &mut reference_keys,
            &mut reference_values,
            &expected_key,
            &expected_value,
            layout,
            0,
            1,
            2,
        );
        reference_copy_tail(&mut reference_keys, &mut reference_values, layout, 1, 2, 3);
        reference_write_table(&mut reference_table, 1, 2);
        let keys = DeviceBuffer::from_host(&device, &initial_keys)?;
        let values = DeviceBuffer::from_host(&device, &initial_values)?;
        let source_key = DeviceBuffer::from_host(&device, &expected_key)?;
        let source_value = DeviceBuffer::from_host(&device, &expected_value)?;
        let table = DeviceBuffer::from_host(&device, &[0_u32, 0])?;
        // SAFETY: the fixture provides distinct descriptor-sized device buffers and retains all of them through synchronization.
        unsafe {
            super::append_row_f32(
                layout,
                keys.as_device_ptr(),
                keys.len(),
                values.as_device_ptr(),
                values.len(),
                source_key.as_device_ptr(),
                source_key.len(),
                source_value.as_device_ptr(),
                source_value.len(),
                0,
                1,
                2,
                &stream,
            )?;
            super::copy_tail_f32(
                layout,
                keys.as_device_ptr(),
                keys.len(),
                values.as_device_ptr(),
                values.len(),
                1,
                2,
                3,
                &stream,
            )?;
            super::write_table_u32(table.as_device_ptr(), table.len(), 1, 2, &stream)?;
        }
        stream.synchronize()?;
        let mut actual_keys = vec![0.0_f32; keys.len()];
        let mut actual_values = vec![0.0_f32; values.len()];
        let mut actual_table = vec![0_u32; table.len()];
        keys.copy_to_host(&mut actual_keys)?;
        values.copy_to_host(&mut actual_values)?;
        table.copy_to_host(&mut actual_table)?;
        assert_eq!(actual_keys, reference_keys);
        assert_eq!(actual_values, reference_values);
        assert_eq!(actual_table, reference_table);
        Ok(())
    }
}
