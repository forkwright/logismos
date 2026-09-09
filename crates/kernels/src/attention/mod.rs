//! Checked single-query paged-attention primitives.
//!
//! The CPU operation accepts logical key/value rows through fallible borrows;
//! it neither owns cache pages nor interprets a cache table. The optional HIP
//! launcher has a separate, explicit dense physical-page descriptor.

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
use hipcore::Stream;
use num_traits::ToPrimitive;
use snafu::{ResultExt, Snafu};

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
use crate::device_span::{
    checked_device_span, checked_f32_device_span, reject_overlapping_device_spans,
};
#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use crate::error::LaunchSnafu;
#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
use crate::error::NoGpuBuildSnafu;
#[cfg(feature = "gpu")]
use crate::error::Result as KernelResult;
#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
use crate::error::UnsupportedShapeSnafu;

const PAGED_DECODE: &str = "paged_decode";
#[cfg(feature = "gpu")]
const PAGED_DECODE_KERNEL: &str = "paged_decode_q1_f32";

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe extern "C" {
    fn logismos_launch_paged_decode_q1_f32(
        query_f32: *const c_void,
        keys_f32: *const c_void,
        values_f32: *const c_void,
        page_table_u32: *const c_void,
        output_f32: *mut c_void,
        visible_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        head_width: u32,
        page_tokens: u32,
        physical_pages: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> u32;
}

/// Result alias for the logical paged-decode operation.
pub type PagedDecodeResult<T> = core::result::Result<T, PagedDecodeError>;

/// Result alias for a logical paged-decode operation with caller-owned rows.
pub type PagedDecodeRowsResult<T, E: std::fmt::Display + std::error::Error + 'static> =
    core::result::Result<T, PagedDecodeRowsError<E>>;

/// Checked concurrent allocation requests for one CPU query-head operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PagedDecodeWorkspace {
    scores: usize,
    head_output: usize,
    total: usize,
}

/// Checked Q=1 logical attention geometry and CPU workspace requirements.
///
/// This plan owns the interpretation of a logical query head, contiguous GQA
/// grouping, one visible causal prefix, and the two live CPU allocations. It
/// deliberately has no page size or physical-table information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedDecodePlan {
    visible_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_width: usize,
    gqa_group: usize,
    query_elements: usize,
    row_elements: usize,
    scale: f32,
    workspace: PagedDecodeWorkspace,
}

impl PagedDecodePlan {
    /// Admit one Q=1 causal visible prefix with contiguous grouped-query heads.
    ///
    /// # Errors
    ///
    /// Returns [`PagedDecodeError`] when a dimension is zero, query heads do
    /// not divide evenly across key/value heads, any complete query/row span
    /// or allocation layout overflows, the f32 scale cannot be represented
    /// finitely, or the required workspace sum overflows.
    pub fn try_from_dimensions(
        visible_tokens: usize,
        query_heads: usize,
        kv_heads: usize,
        head_width: usize,
    ) -> PagedDecodeResult<Self> {
        validate_nonzero_dimension("visible_tokens", visible_tokens)?;
        validate_nonzero_dimension("query_heads", query_heads)?;
        validate_nonzero_dimension("kv_heads", kv_heads)?;
        validate_nonzero_dimension("head_width", head_width)?;
        if !query_heads.is_multiple_of(kv_heads) {
            return HeadGroupingMismatchSnafu {
                query_heads,
                kv_heads,
            }
            .fail();
        }
        let row_elements = checked_product(kv_heads, head_width, "kv_heads * head_width")?;
        let query_elements = checked_product(query_heads, head_width, "query_heads * head_width")?;
        validate_f32_layout("query span", query_elements)?;
        validate_f32_layout("key/value row span", row_elements)?;
        let head_width_f32 = head_width
            .to_f32()
            .filter(|value| value.is_finite())
            .ok_or_else(|| ScaleNotFiniteSnafu { head_width }.build())?;
        let scale = 1.0_f32 / head_width_f32.sqrt();
        ensure_finite(scale, "scale", 0)?;
        let workspace = PagedDecodeWorkspace {
            scores: visible_tokens,
            head_output: head_width,
            total: visible_tokens
                .checked_add(head_width)
                .ok_or_else(|| WorkspaceOverflowSnafu.build())?,
        };
        validate_f32_layout("scores", workspace.scores)?;
        validate_f32_layout("head output", workspace.head_output)?;
        Ok(Self {
            visible_tokens,
            query_heads,
            kv_heads,
            head_width,
            gqa_group: query_heads / kv_heads,
            query_elements,
            row_elements,
            scale,
            workspace,
        })
    }

    /// Return the admitted causal visible-prefix length.
    #[must_use]
    pub const fn visible_tokens(self) -> usize {
        self.visible_tokens
    }

    /// Return the admitted query-head count.
    #[must_use]
    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    /// Return the admitted key/value-head count.
    #[must_use]
    pub const fn kv_heads(self) -> usize {
        self.kv_heads
    }

    /// Return the admitted width of one query/key/value head.
    #[must_use]
    pub const fn head_width(self) -> usize {
        self.head_width
    }

    /// Return the checked contiguous query-heads-per-key/value-head group.
    #[must_use]
    pub const fn gqa_group(self) -> usize {
        self.gqa_group
    }

    /// Return the existing decoder-compatible f32 attention scale.
    #[must_use]
    pub const fn scale(self) -> f32 {
        self.scale
    }

    /// Return the materialized-score allocation required by one CPU head.
    #[must_use]
    pub const fn score_elements(self) -> usize {
        self.workspace.scores
    }

    /// Return the head-output allocation required by one CPU head.
    #[must_use]
    pub const fn head_output_elements(self) -> usize {
        self.workspace.head_output
    }

    /// Return the concurrent CPU score plus head-output allocation request.
    #[must_use]
    pub const fn workspace_elements(self) -> usize {
        self.workspace.total
    }

    fn validate_query(self, query_head: usize, query: &[f32]) -> PagedDecodeResult<usize> {
        if query_head >= self.query_heads {
            return QueryHeadOutOfRangeSnafu {
                query_head,
                query_heads: self.query_heads,
            }
            .fail();
        }
        validate_exact_length("query", query.len(), self.head_width)?;
        validate_finite("query", query)?;
        Ok(query_head / self.gqa_group)
    }

    const fn row_elements(self) -> usize {
        self.row_elements
    }
}

/// Failures while admitting or evaluating logical Q=1 paged attention.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum PagedDecodeError {
    /// A required logical dimension was zero.
    #[snafu(display("{PAGED_DECODE}: {dimension} must be greater than zero"))]
    ZeroDimension {
        /// The rejected dimension.
        dimension: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Query heads could not be grouped contiguously over key/value heads.
    #[snafu(display(
        "{PAGED_DECODE}: query_heads {query_heads} is not divisible by kv_heads {kv_heads}"
    ))]
    HeadGroupingMismatch {
        /// Declared query-head count.
        query_heads: usize,
        /// Declared key/value-head count.
        kv_heads: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A required dimension product could not be represented by `usize`.
    #[snafu(display("{PAGED_DECODE}: {dimensions} overflows usize"))]
    DimensionOverflow {
        /// The product that overflowed.
        dimensions: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The two concurrent CPU workspace allocations could not be summed.
    #[snafu(display("{PAGED_DECODE}: score and output workspace sum overflows usize"))]
    WorkspaceOverflow {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An exact span or allocation extent cannot form a Rust layout.
    #[snafu(display(
        "{PAGED_DECODE}: {allocation} layout for {elements} elements is unrepresentable"
    ))]
    AllocationLayout {
        /// Span or allocation role.
        allocation: &'static str,
        /// Exact requested element count.
        elements: usize,
        /// Layout failure reported by the standard library.
        source: std::alloc::LayoutError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The head width could not yield a finite f32 attention scale.
    #[snafu(display("{PAGED_DECODE}: head_width {head_width} has no finite f32 scale"))]
    ScaleNotFinite {
        /// The rejected head width.
        head_width: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A native physical-page size is outside this operation's explicit ABI.
    #[snafu(display(
        "{PAGED_DECODE}: native page_tokens {page_tokens} is not one of B8, B16, or B32"
    ))]
    NativePageTokensUnsupported {
        /// Rejected native physical-page token count.
        page_tokens: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A checked native descriptor dimension cannot cross the u32 HIP ABI.
    #[snafu(display("{PAGED_DECODE}: native {dimension} {value} exceeds the HIP ABI u32 domain"))]
    NativeAbiOutOfRange {
        /// Native descriptor dimension that failed conversion.
        dimension: &'static str,
        /// Rejected native descriptor dimension value.
        value: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A requested query head was outside the admitted query-head range.
    #[snafu(display("{PAGED_DECODE}: query head {query_head} is outside 0..{query_heads}"))]
    QueryHeadOutOfRange {
        /// Requested query-head index.
        query_head: usize,
        /// Admitted query-head count.
        query_heads: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A logical input or borrowed row did not have its exact admitted width.
    #[snafu(display("{PAGED_DECODE}: {input} length {actual} does not match expected {expected}"))]
    LengthMismatch {
        /// Input or row role that failed validation.
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
    #[snafu(display("{PAGED_DECODE}: {input}[{index}] is not finite"))]
    NonFiniteInput {
        /// Input or row role containing the scalar.
        input: &'static str,
        /// Flat scalar index.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A score, normalization, or output intermediate was not finite.
    #[snafu(display("{PAGED_DECODE}: non-finite value during {stage} at index {index}"))]
    NonFiniteArithmetic {
        /// Named evaluation stage.
        stage: &'static str,
        /// Logical token or output-coordinate index.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// One exact CPU workspace allocation could not be reserved.
    #[snafu(display("{PAGED_DECODE}: could not reserve {elements} f32 elements for {allocation}"))]
    Allocation {
        /// Logical workspace allocation role.
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

/// A typed failure from the logical operation or one caller-owned row borrow.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum PagedDecodeRowsError<E: std::fmt::Display + std::error::Error + 'static> {
    /// The checked operation rejected its geometry, input, or arithmetic.
    #[snafu(display("{PAGED_DECODE}: operation failure: {source}"))]
    Kernel {
        /// Source operation failure.
        source: PagedDecodeError,
    },

    /// The caller could not borrow a key row at one logical token.
    #[snafu(display("{PAGED_DECODE}: key row {token} unavailable: {source}"))]
    KeyRow {
        /// Logical visible-token index.
        token: usize,
        /// Typed caller-owned row-borrow failure.
        source: E,
    },

    /// The caller could not borrow a value row at one logical token.
    #[snafu(display("{PAGED_DECODE}: value row {token} unavailable: {source}"))]
    ValueRow {
        /// Logical visible-token index.
        token: usize,
        /// Typed caller-owned row-borrow failure.
        source: E,
    },
}

/// Evaluate one query head over a caller-owned logical causal visible prefix.
///
/// `key_row` and `value_row` each borrow one row in ascending logical-token
/// order. A row is exactly `[kv_heads, head_width]`; the selected key/value
/// head is `query_head / gqa_group`. The f32 calculation intentionally keeps
/// the decoder's materialized-score order: scores, then maximum, then
/// normalizer, then output values all progress in ascending token order. It
/// validates every full-row extent but inspects finite scalars only in the
/// selected head, so unrelated GQA heads are not operation inputs.
///
/// # Errors
///
/// Returns [`PagedDecodeRowsError::Kernel`] for plan, shape, allocation, or
/// finite-domain failures. A row provider's original typed error remains the
/// source of [`PagedDecodeRowsError::KeyRow`] or
/// [`PagedDecodeRowsError::ValueRow`].
pub fn paged_decode_cpu<'rows, E, KeyRow, ValueRow>(
    plan: PagedDecodePlan,
    query_head: usize,
    query: &[f32],
    mut key_row: KeyRow,
    mut value_row: ValueRow,
) -> PagedDecodeRowsResult<Vec<f32>, E>
where
    E: std::fmt::Display + std::error::Error + 'static,
    KeyRow: FnMut(usize) -> core::result::Result<&'rows [f32], E>,
    ValueRow: FnMut(usize) -> core::result::Result<&'rows [f32], E>,
{
    let kv_head = plan
        .validate_query(query_head, query)
        .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
    let row_elements = plan.row_elements();
    let head_start =
        kv_head
            .checked_mul(plan.head_width)
            .ok_or_else(|| PagedDecodeRowsError::Kernel {
                source: DimensionOverflowSnafu {
                    dimensions: "kv_head * head_width",
                }
                .build(),
            })?;
    let head_end =
        head_start
            .checked_add(plan.head_width)
            .ok_or_else(|| PagedDecodeRowsError::Kernel {
                source: DimensionOverflowSnafu {
                    dimensions: "selected key/value head range",
                }
                .build(),
            })?;

    let mut scores = reserve_f32("scores", plan.score_elements())
        .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
    for token in 0..plan.visible_tokens {
        let row =
            key_row(token).map_err(|source| PagedDecodeRowsError::KeyRow { token, source })?;
        validate_exact_length("key row", row.len(), row_elements)
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        validate_finite("selected key head", &row[head_start..head_end])
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        let score = scaled_dot(query, &row[head_start..head_end], plan.scale, token)
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        scores.push(score);
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ensure_finite(max_score, "score maximum", 0)
        .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
    let mut normalizer = 0.0_f32;
    for (token, score) in scores.iter().copied().enumerate() {
        let weight = (score - max_score).exp();
        ensure_finite(weight, "score exponent", token)
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        normalizer += weight;
        ensure_finite(normalizer, "score normalizer", token)
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
    }
    if normalizer == 0.0_f32 {
        return Err(PagedDecodeRowsError::Kernel {
            source: NonFiniteArithmeticSnafu {
                stage: "score normalizer",
                index: 0_usize,
            }
            .build(),
        });
    }

    let mut output = reserve_f32("head output", plan.head_output_elements())
        .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
    output.resize(plan.head_output_elements(), 0.0_f32);
    for (token, score) in scores.iter().copied().enumerate() {
        let probability = (score - max_score).exp() / normalizer;
        ensure_finite(probability, "score probability", token)
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        let row =
            value_row(token).map_err(|source| PagedDecodeRowsError::ValueRow { token, source })?;
        validate_exact_length("value row", row.len(), row_elements)
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        validate_finite("selected value head", &row[head_start..head_end])
            .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        for (column, value) in row[head_start..head_end].iter().copied().enumerate() {
            output[column] += probability * value;
            ensure_finite(output[column], "head output", column)
                .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
        }
    }
    Ok(output)
}

fn scaled_dot(query: &[f32], key: &[f32], scale: f32, token: usize) -> PagedDecodeResult<f32> {
    let mut dot = 0.0_f32;
    for (column, (query_value, key_value)) in query.iter().zip(key).enumerate() {
        let product = *query_value * *key_value;
        ensure_finite(product, "score product", column)?;
        dot += product;
        ensure_finite(dot, "score dot", column)?;
    }
    let score = dot * scale;
    ensure_finite(score, "score", token)?;
    Ok(score)
}

fn reserve_f32(allocation: &'static str, elements: usize) -> PagedDecodeResult<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .context(AllocationSnafu {
            allocation,
            elements,
        })?;
    Ok(values)
}

fn validate_f32_layout(allocation: &'static str, elements: usize) -> PagedDecodeResult<()> {
    validate_layout::<f32>(allocation, elements)
}

fn validate_layout<T>(allocation: &'static str, elements: usize) -> PagedDecodeResult<()> {
    std::alloc::Layout::array::<T>(elements).context(AllocationLayoutSnafu {
        allocation,
        elements,
    })?;
    Ok(())
}

fn validate_nonzero_dimension(dimension: &'static str, value: usize) -> PagedDecodeResult<()> {
    if value == 0 {
        return ZeroDimensionSnafu { dimension }.fail();
    }
    Ok(())
}

fn validate_exact_length(
    input: &'static str,
    actual: usize,
    expected: usize,
) -> PagedDecodeResult<()> {
    if actual == expected {
        Ok(())
    } else {
        LengthMismatchSnafu {
            input,
            expected,
            actual,
        }
        .fail()
    }
}

fn validate_finite(input: &'static str, values: &[f32]) -> PagedDecodeResult<()> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return NonFiniteInputSnafu { input, index }.fail();
        }
    }
    Ok(())
}

fn ensure_finite(value: f32, stage: &'static str, index: usize) -> PagedDecodeResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        NonFiniteArithmeticSnafu { stage, index }.fail()
    }
}

/// Explicit physical tokens per native dense page.
///
/// This selection belongs only to the native addressing descriptor. It does
/// not reuse or imply the CPU cache allocation selector.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativePageTokens {
    /// Eight logical tokens per physical native page.
    B8,
    /// Sixteen logical tokens per physical native page.
    B16,
    /// Thirty-two logical tokens per physical native page.
    B32,
}

#[cfg(feature = "gpu")]
impl NativePageTokens {
    /// Return the explicit number of logical tokens in one physical page.
    #[must_use]
    pub const fn get(self) -> usize {
        match self {
            Self::B8 => 8,
            Self::B16 => 16,
            Self::B32 => 32,
        }
    }

    fn try_from_tokens(page_tokens: usize) -> PagedDecodeResult<Self> {
        match page_tokens {
            8 => Ok(Self::B8),
            16 => Ok(Self::B16),
            32 => Ok(Self::B32),
            _ => NativePageTokensUnsupportedSnafu { page_tokens }.fail(),
        }
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativePagedDecodeAbi {
    visible_tokens: u32,
    query_heads: u32,
    kv_heads: u32,
    head_width: u32,
    page_tokens: u32,
    physical_pages: u32,
}

/// Checked native address layout for one all-query-head Q=1 decode call.
///
/// Keys and values are separate dense `f32` arrays with logical layout
/// `[physical_page][in_page_token][kv_head][head_width]`. `page_table` has
/// one `u32` physical-page index per logical page. The descriptor eagerly
/// checks dense K/V and table allocation layouts plus ABI dimensions, but
/// cannot inspect device table values or scalar contents.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativePagedDecodePlan {
    logical: PagedDecodePlan,
    page_tokens: NativePageTokens,
    physical_pages: usize,
    logical_pages: usize,
    query_elements: usize,
    key_value_elements: usize,
    output_elements: usize,
    abi: NativePagedDecodeAbi,
}

#[cfg(feature = "gpu")]
impl NativePagedDecodePlan {
    /// Derive one explicit native dense-page descriptor from logical geometry.
    ///
    /// # Errors
    ///
    /// Returns [`PagedDecodeError`] when the requested native page size is not
    /// B8/B16/B32, physical-page count is zero, a native extent overflows, or
    /// a dense K/V or table allocation layout or any u32 ABI dimension is
    /// unrepresentable.
    pub fn try_from_paged_decode(
        logical: PagedDecodePlan,
        page_tokens: usize,
        physical_pages: usize,
    ) -> PagedDecodeResult<Self> {
        let page_tokens = NativePageTokens::try_from_tokens(page_tokens)?;
        validate_nonzero_dimension("physical_pages", physical_pages)?;
        let abi = NativePagedDecodeAbi {
            visible_tokens: native_abi_u32("visible_tokens", logical.visible_tokens)?,
            kv_heads: native_abi_u32("kv_heads", logical.kv_heads)?,
            query_heads: native_abi_u32("query_heads", logical.query_heads)?,
            head_width: native_abi_u32("head_width", logical.head_width)?,
            page_tokens: native_abi_u32("page_tokens", page_tokens.get())?,
            physical_pages: native_abi_u32("physical_pages", physical_pages)?,
        };
        let logical_pages = logical
            .visible_tokens
            .checked_sub(1)
            .and_then(|last| last.checked_div(page_tokens.get()))
            .and_then(|page| page.checked_add(1))
            .ok_or_else(|| {
                DimensionOverflowSnafu {
                    dimensions: "visible_tokens / page_tokens",
                }
                .build()
            })?;
        let query_elements = logical.query_elements;
        let key_value_elements = checked_product(
            checked_product(
                physical_pages,
                page_tokens.get(),
                "physical_pages * page_tokens",
            )?,
            logical.row_elements(),
            "physical_pages * page_tokens * kv_heads * head_width",
        )?;
        validate_f32_layout("native keys", key_value_elements)?;
        validate_f32_layout("native values", key_value_elements)?;
        validate_layout::<u32>("native page table", logical_pages)?;
        Ok(Self {
            logical,
            page_tokens,
            physical_pages,
            logical_pages,
            query_elements,
            key_value_elements,
            output_elements: query_elements,
            abi,
        })
    }

    /// Return the logical Q=1 geometry that this native descriptor addresses.
    #[must_use]
    pub const fn logical(self) -> PagedDecodePlan {
        self.logical
    }

    /// Return this descriptor's explicit physical page-token selector.
    #[must_use]
    pub const fn page_tokens(self) -> NativePageTokens {
        self.page_tokens
    }

    /// Return the supplied dense physical-page count.
    #[must_use]
    pub const fn physical_pages(self) -> usize {
        self.physical_pages
    }

    /// Return the checked table entry count for the visible logical prefix.
    #[must_use]
    pub const fn page_table_entries(self) -> usize {
        self.logical_pages
    }

    /// Return the dense query extent `[query_heads, head_width]`.
    #[must_use]
    pub const fn query_elements(self) -> usize {
        self.query_elements
    }

    /// Return the separate dense key or value physical-page extent.
    #[must_use]
    pub const fn key_value_elements(self) -> usize {
        self.key_value_elements
    }

    /// Return the dense output extent `[query_heads, head_width]`.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output_elements
    }
}

/// Launch a staged all-query-head native Q=1 paged-attention operation.
///
/// The native descriptor is distinct from logical CPU cache ownership. Key
/// and value inputs use `[physical_page][in_page_token][kv_head][head_width]`
/// with separate arrays; each `u32` table entry maps one logical page to a
/// physical page. `query_f32` and `output_f32` are `[query_head][head_width]`.
/// The plan's contiguous GQA rule selects `query_head / gqa_group`.
///
/// The one-wave-per-query-head kernel uses all 32 lanes, including when the
/// width is below 32 or has a tail. A lane serially accumulates coordinates
/// `lane, lane + 32, ...`; a fixed down-shuffle tree (16, 8, 4, 2, 1) produces
/// one dot product, and lane zero applies the f32 scale once. Every lane then
/// updates its strided coordinates in `output_f32` as unnormalized running
/// output while retaining the same online maximum and normalizer. Final
/// division by the normalizer publishes each coordinate. This is mathematically
/// equivalent to softmax but intentionally not the CPU materialized rounding
/// order. The native source disables contraction and reassociation.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] when a supplied extent differs
/// from the descriptor, a nonempty span is null, unaligned, unrepresentable,
/// or aliases the writable output. Descriptor construction already rejects
/// dimensions outside the `u32` ABI. A CPU-only build returns
/// [`crate::Error::NoGpuBuild`] without initializing HIP. HIP stream-current
/// and submission failures are propagated.
///
/// # Safety
///
/// Each nonempty pointer must remain live on `stream`'s device through stream
/// completion. The output span requires exclusive access and must not alias
/// any input. The caller must ensure every table entry is less than
/// `physical_pages`, all inputs and every online intermediate are finite and
/// normal-or-zero, and the selected device supports the fixed 32-thread block.
/// Device contents are not inspectable here, so this boundary cannot reproduce
/// CPU row, table-value, or finite-domain refusals.
#[cfg(feature = "gpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "the five native buffers, declared extents, descriptor, and stream form the fixed Q=1 ABI"
)]
pub unsafe fn launch_paged_decode_q1_f32(
    plan: NativePagedDecodePlan,
    query_f32: *const f32,
    query_elements: usize,
    keys_f32: *const f32,
    key_elements: usize,
    values_f32: *const f32,
    value_elements: usize,
    page_table_u32: *const u32,
    page_table_entries: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
) -> KernelResult<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            query_f32,
            query_elements,
            keys_f32,
            key_elements,
            values_f32,
            value_elements,
            page_table_u32,
            page_table_entries,
            output_f32,
            output_elements,
            stream,
        );
        no_gpu_paged_decode_refusal()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let abi = validate_paged_decode_launch(
            plan,
            query_f32,
            query_elements,
            keys_f32,
            key_elements,
            values_f32,
            value_elements,
            page_table_u32,
            page_table_entries,
            output_f32,
            output_elements,
        )?;
        stream.make_current()?;
        // SAFETY: the caller upholds device ownership, lifetime, concurrent
        // access, table-value, and numerical-domain obligations documented
        // above; descriptor and span validation establish ABI extents.
        let code = unsafe {
            logismos_launch_paged_decode_q1_f32(
                query_f32.cast::<c_void>(),
                keys_f32.cast::<c_void>(),
                values_f32.cast::<c_void>(),
                page_table_u32.cast::<c_void>(),
                output_f32.cast::<c_void>(),
                abi.visible_tokens,
                abi.query_heads,
                abi.kv_heads,
                abi.head_width,
                abi.page_tokens,
                abi.physical_pages,
                plan.logical.scale,
                stream.raw().cast::<c_void>(),
            )
        };
        if code == 0 {
            Ok(())
        } else {
            LaunchSnafu {
                kernel: PAGED_DECODE_KERNEL,
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            }
            .fail()
        }
    }
}

#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
fn no_gpu_paged_decode_refusal() -> KernelResult<()> {
    NoGpuBuildSnafu {
        kernel: PAGED_DECODE_KERNEL,
    }
    .fail()
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
#[expect(
    clippy::too_many_arguments,
    reason = "validation receives the fixed raw Q=1 paged-attention ABI without a second geometry owner"
)]
fn validate_paged_decode_launch(
    plan: NativePagedDecodePlan,
    query_f32: *const f32,
    query_elements: usize,
    keys_f32: *const f32,
    key_elements: usize,
    values_f32: *const f32,
    value_elements: usize,
    page_table_u32: *const u32,
    page_table_entries: usize,
    output_f32: *mut f32,
    output_elements: usize,
) -> KernelResult<NativePagedDecodeAbi> {
    validate_native_length("query", query_elements, plan.query_elements)?;
    validate_native_length("keys", key_elements, plan.key_value_elements)?;
    validate_native_length("values", value_elements, plan.key_value_elements)?;
    validate_native_length("page table", page_table_entries, plan.logical_pages)?;
    validate_native_length("output", output_elements, plan.output_elements)?;

    let query = checked_f32_device_span(PAGED_DECODE_KERNEL, query_f32, query_elements, "query")?;
    let keys = checked_f32_device_span(PAGED_DECODE_KERNEL, keys_f32, key_elements, "keys")?;
    let values =
        checked_f32_device_span(PAGED_DECODE_KERNEL, values_f32, value_elements, "values")?;
    let table = checked_device_span(
        PAGED_DECODE_KERNEL,
        page_table_u32,
        page_table_entries,
        "page table",
    )?;
    let output = checked_f32_device_span(
        PAGED_DECODE_KERNEL,
        output_f32.cast_const(),
        output_elements,
        "output",
    )?;
    for input in [query, keys, values, table] {
        reject_overlapping_device_spans(PAGED_DECODE_KERNEL, output, input)?;
    }

    Ok(plan.abi)
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn validate_native_length(name: &'static str, actual: usize, expected: usize) -> KernelResult<()> {
    if actual == expected {
        Ok(())
    } else {
        UnsupportedShapeSnafu {
            kernel: PAGED_DECODE_KERNEL,
            msg: format!(
                "{name} length {actual} does not match native descriptor extent {expected}"
            ),
        }
        .fail()
    }
}

#[cfg(feature = "gpu")]
fn native_abi_u32(dimension: &'static str, value: usize) -> PagedDecodeResult<u32> {
    u32::try_from(value).map_err(|_| NativeAbiOutOfRangeSnafu { dimension, value }.build())
}

fn checked_product(
    left: usize,
    right: usize,
    dimensions: &'static str,
) -> PagedDecodeResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| DimensionOverflowSnafu { dimensions }.build())
}

#[cfg(all(test, feature = "gpu"))]
mod native_tests;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn materialized_cpu_matches_independent_f64_gqa_oracle()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = PagedDecodePlan::try_from_dimensions(3, 4, 2, 3)?;
        let query = [0.25_f32, -0.5, 1.0];
        let keys = [
            vec![2.0, 0.0, 0.0, 0.25, -0.5, 1.0],
            vec![0.0, 2.0, 0.0, 0.5, -0.25, 0.75],
            vec![0.0, 0.0, 2.0, -0.5, 0.25, 1.25],
        ];
        let values = [
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            vec![4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            vec![7.0, 8.0, 9.0, 70.0, 80.0, 90.0],
        ];

        let actual = paged_decode_cpu(
            plan,
            2,
            &query,
            |token| Ok::<_, std::io::Error>(keys[token].as_slice()),
            |token| Ok::<_, std::io::Error>(values[token].as_slice()),
        )?;
        let expected = f64_materialized_oracle(&query, 1, &keys, &values)?;

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_relative_eq!(*actual, expected as f32, epsilon = 1.0e-5);
        }
        assert_eq!(plan.gqa_group(), 2);
        assert_eq!(plan.score_elements(), 3);
        assert_eq!(plan.head_output_elements(), 3);
        assert_eq!(plan.workspace_elements(), 6);
        Ok(())
    }

    #[test]
    fn materialized_cpu_observes_ascending_visible_prefix_order()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = PagedDecodePlan::try_from_dimensions(2, 1, 1, 1)?;
        let query = [1.0_f32];
        let keys = [vec![0.0_f32], vec![1.0]];
        let values = [vec![3.0_f32], vec![11.0]];
        let mut key_tokens = Vec::new();
        let mut value_tokens = Vec::new();

        let output = paged_decode_cpu(
            plan,
            0,
            &query,
            |token| {
                key_tokens.push(token);
                Ok::<_, std::io::Error>(keys[token].as_slice())
            },
            |token| {
                value_tokens.push(token);
                Ok::<_, std::io::Error>(values[token].as_slice())
            },
        )?;

        assert_eq!(key_tokens, [0, 1]);
        assert_eq!(value_tokens, [0, 1]);
        let expected = (3.0_f64 + 11.0 * 1.0_f64.exp()) / (1.0 + 1.0_f64.exp());
        assert_relative_eq!(output[0], expected as f32, epsilon = 1.0e-5);
        Ok(())
    }

    #[test]
    fn row_borrow_and_shape_errors_remain_typed()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = PagedDecodePlan::try_from_dimensions(2, 1, 1, 2)?;
        let query = [1.0_f32, 2.0];
        let row = [0.0_f32, 1.0];
        let key_failure = paged_decode_cpu(
            plan,
            0,
            &query,
            |_| Err::<&[f32], _>(std::io::Error::other("key unavailable")),
            |_| Ok::<_, std::io::Error>(&row),
        );
        assert!(matches!(
            key_failure,
            Err(PagedDecodeRowsError::KeyRow { token: 0, .. })
        ));

        let short = [0.0_f32];
        let row_shape = paged_decode_cpu(
            plan,
            0,
            &query,
            |_| Ok::<_, std::io::Error>(&short),
            |_| Ok::<_, std::io::Error>(&row),
        );
        assert!(matches!(
            row_shape,
            Err(PagedDecodeRowsError::Kernel {
                source: PagedDecodeError::LengthMismatch {
                    input: "key row",
                    ..
                }
            })
        ));
        Ok(())
    }

    #[test]
    fn plan_and_cpu_refuse_invalid_geometry_and_nonfinite_inputs()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        assert!(matches!(
            PagedDecodePlan::try_from_dimensions(1, 3, 2, 1),
            Err(PagedDecodeError::HeadGroupingMismatch { .. })
        ));
        assert!(matches!(
            PagedDecodePlan::try_from_dimensions(0, 1, 1, 1),
            Err(PagedDecodeError::ZeroDimension {
                dimension: "visible_tokens",
                ..
            })
        ));

        let plan = PagedDecodePlan::try_from_dimensions(1, 1, 1, 1)?;
        let nonfinite = [f32::NAN];
        let row = [0.0_f32];
        let result = paged_decode_cpu(
            plan,
            0,
            &nonfinite,
            |_| Ok::<_, std::io::Error>(&row),
            |_| Ok::<_, std::io::Error>(&row),
        );
        assert!(matches!(
            result,
            Err(PagedDecodeRowsError::Kernel {
                source: PagedDecodeError::NonFiniteInput {
                    input: "query",
                    index: 0,
                    ..
                }
            })
        ));
        Ok(())
    }

    #[test]
    fn constructor_eagerly_refuses_complete_span_and_allocation_layout_overflow() {
        assert!(matches!(
            PagedDecodePlan::try_from_dimensions(1, usize::MAX, usize::MAX, 2),
            Err(PagedDecodeError::DimensionOverflow {
                dimensions: "kv_heads * head_width",
                ..
            })
        ));
        assert!(matches!(
            PagedDecodePlan::try_from_dimensions(1, usize::MAX, 1, 2),
            Err(PagedDecodeError::DimensionOverflow {
                dimensions: "query_heads * head_width",
                ..
            })
        ));
        let layout_overflow = (isize::MAX as usize / core::mem::size_of::<f32>()) + 1;
        assert!(matches!(
            PagedDecodePlan::try_from_dimensions(layout_overflow, 1, 1, 1),
            Err(PagedDecodeError::AllocationLayout {
                allocation: "scores",
                elements,
                ..
            }) if elements == layout_overflow
        ));
    }

    #[test]
    fn per_head_operation_ignores_other_gqa_heads_but_refuses_selected_poison()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = PagedDecodePlan::try_from_dimensions(1, 2, 2, 2)?;
        let query = [1.0_f32, 0.0];
        let unused_poison = [1.0_f32, 0.0, f32::NAN, f32::NAN];
        let values = [3.0_f32, 5.0, f32::NAN, f32::NAN];
        let output = paged_decode_cpu(
            plan,
            0,
            &query,
            |_| Ok::<_, std::io::Error>(&unused_poison),
            |_| Ok::<_, std::io::Error>(&values),
        )?;
        assert_eq!(output.as_slice(), &[3.0, 5.0]);

        let selected_poison = [f32::NAN, 0.0, 1.0, 0.0];
        let refused = paged_decode_cpu(
            plan,
            0,
            &query,
            |_| Ok::<_, std::io::Error>(&selected_poison),
            |_| Ok::<_, std::io::Error>(&values),
        );
        assert!(matches!(
            refused,
            Err(PagedDecodeRowsError::Kernel {
                source: PagedDecodeError::NonFiniteInput {
                    input: "selected key head",
                    index: 0,
                    ..
                }
            })
        ));
        Ok(())
    }

    #[test]
    fn independent_online_f64_witness_matches_actual_cpu_with_repeated_maximums()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let query = [1.0_f32, -1.0];
        let keys = [
            vec![3.0, -3.0],
            vec![3.0, -3.0],
            vec![-2.0, 2.0],
            vec![1.0, -1.0],
        ];
        let values = [
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![4.0, 40.0],
            vec![8.0, 80.0],
        ];
        let materialized = f64_materialized_oracle(&query, 0, &keys, &values)?;
        let online = f64_online_oracle(&query, 0, &keys, &values)?;
        let plan = PagedDecodePlan::try_from_dimensions(keys.len(), 1, 1, query.len())?;
        let actual = paged_decode_cpu(
            plan,
            0,
            &query,
            |token| Ok::<_, std::io::Error>(keys[token].as_slice()),
            |token| Ok::<_, std::io::Error>(values[token].as_slice()),
        )?;

        for ((materialized, online), actual) in materialized.iter().zip(online).zip(actual) {
            assert_relative_eq!(*materialized, online, epsilon = 1.0e-12);
            assert_relative_eq!(actual, *materialized as f32, epsilon = 1.0e-5);
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn native_descriptor_is_explicit_and_span_validation_refuses_bad_extents()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let logical = PagedDecodePlan::try_from_dimensions(9, 2, 1, 3)?;
        let native = NativePagedDecodePlan::try_from_paged_decode(logical, 8, 4)?;
        assert_eq!(native.page_tokens(), NativePageTokens::B8);
        assert_eq!(native.page_table_entries(), 2);
        assert_eq!(native.query_elements(), 6);
        assert_eq!(native.key_value_elements(), 96);
        assert_eq!(native.output_elements(), 6);
        assert_eq!(
            NativePagedDecodePlan::try_from_paged_decode(logical, 16, 1)?.page_tokens(),
            NativePageTokens::B16
        );
        assert_eq!(
            NativePagedDecodePlan::try_from_paged_decode(logical, 32, 1)?.page_tokens(),
            NativePageTokens::B32
        );
        assert!(matches!(
            NativePagedDecodePlan::try_from_paged_decode(logical, 12, 1),
            Err(PagedDecodeError::NativePageTokensUnsupported { .. })
        ));
        let largest_abi_page_count = usize::try_from(u32::MAX)?;
        let native_layout_width = isize::MAX as usize
            / core::mem::size_of::<f32>()
            / NativePageTokens::B8.get()
            / largest_abi_page_count
            + 1;
        let native_layout_logical =
            PagedDecodePlan::try_from_dimensions(1, 1, 1, native_layout_width)?;
        assert!(matches!(
            NativePagedDecodePlan::try_from_paged_decode(
                native_layout_logical,
                NativePageTokens::B8.get(),
                largest_abi_page_count,
            ),
            Err(PagedDecodeError::AllocationLayout {
                allocation: "native keys",
                ..
            })
        ));
        let beyond_u32 = usize::try_from(u32::MAX)?
            .checked_add(1)
            .ok_or("host usize cannot represent the u32 ABI boundary")?;
        let visible_beyond_abi = PagedDecodePlan::try_from_dimensions(beyond_u32, 1, 1, 1)?;
        assert!(matches!(
            NativePagedDecodePlan::try_from_paged_decode(visible_beyond_abi, 8, 1),
            Err(PagedDecodeError::NativeAbiOutOfRange {
                dimension: "visible_tokens",
                value,
                ..
            }) if value == beyond_u32
        ));
        let query_heads_beyond_abi = PagedDecodePlan::try_from_dimensions(1, beyond_u32, 1, 1)?;
        assert!(matches!(
            NativePagedDecodePlan::try_from_paged_decode(query_heads_beyond_abi, 8, 1),
            Err(PagedDecodeError::NativeAbiOutOfRange {
                dimension: "query_heads",
                value,
                ..
            }) if value == beyond_u32
        ));
        let kv_heads_beyond_abi =
            PagedDecodePlan::try_from_dimensions(1, beyond_u32, beyond_u32, 1)?;
        assert!(matches!(
            NativePagedDecodePlan::try_from_paged_decode(kv_heads_beyond_abi, 8, 1),
            Err(PagedDecodeError::NativeAbiOutOfRange {
                dimension: "kv_heads",
                value,
                ..
            }) if value == beyond_u32
        ));
        let width_beyond_abi = PagedDecodePlan::try_from_dimensions(1, 1, 1, beyond_u32)?;
        assert!(matches!(
            NativePagedDecodePlan::try_from_paged_decode(width_beyond_abi, 8, 1),
            Err(PagedDecodeError::NativeAbiOutOfRange {
                dimension: "head_width",
                value,
                ..
            }) if value == beyond_u32
        ));
        assert!(matches!(
            NativePagedDecodePlan::try_from_paged_decode(
                PagedDecodePlan::try_from_dimensions(1, 1, 1, 1)?,
                8,
                beyond_u32,
            ),
            Err(PagedDecodeError::NativeAbiOutOfRange {
                dimension: "physical_pages",
                value,
                ..
            }) if value == beyond_u32
        ));

        let query = 0x1000_usize as *const f32;
        let keys = 0x2000_usize as *const f32;
        let values = 0x3000_usize as *const f32;
        let table = 0x4000_usize as *const u32;
        let output = 0x5000_usize as *mut f32;
        assert!(
            validate_paged_decode_launch(
                native,
                query,
                native.query_elements(),
                keys,
                native.key_value_elements(),
                values,
                native.key_value_elements(),
                table,
                native.page_table_entries(),
                output,
                native.output_elements(),
            )
            .is_ok()
        );
        assert!(matches!(
            validate_paged_decode_launch(
                native,
                query,
                native.query_elements(),
                keys,
                native.key_value_elements(),
                values,
                native.key_value_elements(),
                table,
                native.page_table_entries(),
                output,
                native.output_elements() - 1,
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn wide_native_coordinate_stride_reaches_max_admitted_width_without_wrap()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        const WAVE_WIDTH: u64 = 32;
        let width = u64::from(u32::MAX);
        let max_width = usize::try_from(u32::MAX)?;
        let logical = PagedDecodePlan::try_from_dimensions(1, 1, 1, max_width)?;
        let native = NativePagedDecodePlan::try_from_paged_decode(logical, 8, 1)?;
        assert_eq!(native.logical().head_width(), max_width);
        for lane in [0_u64, WAVE_WIDTH - 1] {
            let last = lane + ((width - 1 - lane) / WAVE_WIDTH) * WAVE_WIDTH;
            let next = last
                .checked_add(WAVE_WIDTH)
                .ok_or("wide native coordinate stride overflowed")?;
            assert!(last < width);
            assert!(next >= width);
        }
        Ok(())
    }

    fn f64_materialized_oracle(
        query: &[f32],
        kv_head: usize,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> core::result::Result<Vec<f64>, Box<dyn std::error::Error>> {
        let width = query.len();
        let scale = 1.0_f64 / (width as f64).sqrt();
        let head_start = kv_head * width;
        let head_end = head_start + width;
        let mut scores = Vec::new();
        scores.try_reserve_exact(keys.len())?;
        for key_row in keys {
            let mut dot = 0.0_f64;
            for (query_value, key_value) in query.iter().zip(&key_row[head_start..head_end]) {
                dot += f64::from(*query_value) * f64::from(*key_value);
            }
            scores.push(dot * scale);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let normalizer: f64 = scores.iter().map(|score| (score - maximum).exp()).sum();
        let mut output = vec![0.0_f64; width];
        for (token, score) in scores.iter().copied().enumerate() {
            let probability = (score - maximum).exp() / normalizer;
            for (column, value) in values[token][head_start..head_end]
                .iter()
                .copied()
                .enumerate()
            {
                output[column] += probability * f64::from(value);
            }
        }
        Ok(output)
    }

    fn f64_online_oracle(
        query: &[f32],
        kv_head: usize,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> core::result::Result<Vec<f64>, Box<dyn std::error::Error>> {
        let width = query.len();
        let scale = 1.0_f64 / (width as f64).sqrt();
        let head_start = kv_head * width;
        let head_end = head_start + width;
        let mut maximum = f64::NEG_INFINITY;
        let mut normalizer = 0.0_f64;
        let mut output = vec![0.0_f64; width];
        for (key_row, value_row) in keys.iter().zip(values) {
            let mut dot = 0.0_f64;
            for (query_value, key_value) in query.iter().zip(&key_row[head_start..head_end]) {
                dot += f64::from(*query_value) * f64::from(*key_value);
            }
            let score = dot * scale;
            let next_maximum = maximum.max(score);
            let prior_rescale = if maximum.is_infinite() {
                0.0_f64
            } else {
                (maximum - next_maximum).exp()
            };
            let token_weight = (score - next_maximum).exp();
            for (result, value) in output.iter_mut().zip(&value_row[head_start..head_end]) {
                *result = prior_rescale * *result + token_weight * f64::from(*value);
            }
            maximum = next_maximum;
            normalizer = prior_rescale * normalizer + token_weight;
        }
        for result in &mut output {
            *result /= normalizer;
        }
        Ok(output)
    }
}
