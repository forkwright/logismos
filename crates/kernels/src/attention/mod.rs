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
#[cfg(feature = "gpu")]
use crate::numerical_status::NativeNumericalStatus;
use crate::packed_prefill::PackedPrefillPlan;

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
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
}

/// Result alias for the logical paged-decode operation.
pub type PagedDecodeResult<T> = core::result::Result<T, PagedDecodeError>;

/// Result alias for a logical paged-decode operation with caller-owned rows.
pub type PagedDecodeRowsResult<T, E> = core::result::Result<T, PagedDecodeRowsError<E>>;

/// Result alias for the logical single-sequence paged-prefill operation.
pub type PagedPrefillResult<T> = core::result::Result<T, PagedPrefillError>;

/// Result alias for paged prefill with caller-owned key/value rows.
pub type PagedPrefillRowsResult<T, E> = core::result::Result<T, PagedPrefillRowsError<E>>;

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
pub enum PagedDecodeRowsError<E: std::error::Error + 'static> {
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

/// Checked B=1 causal attention geometry for one packed model chunk.
///
/// Construction borrows the packed plan as the authority for token count and
/// committed offset, then lowers those facts into this owned descriptor. The
/// descriptor adds the attention axes and active query/output spans
/// `[tokens, query_heads, head_width]` without retaining a second position
/// authority.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedPrefillPlan {
    last_query: PagedDecodePlan,
    tokens: usize,
    query_elements: usize,
}

impl PagedPrefillPlan {
    /// Derive B=1 causal attention geometry from one borrowed packed chunk.
    ///
    /// # Errors
    ///
    /// Returns [`PagedPrefillError`] when the packed operation contains any
    /// batch other than one, attention dimensions are invalid, or an active
    /// query/output extent cannot be represented.
    pub fn try_from_packed_prefill(
        packed: &PackedPrefillPlan,
        query_heads: usize,
        kv_heads: usize,
        head_width: usize,
    ) -> PagedPrefillResult<Self> {
        if packed.sequence_count() != 1 {
            return BatchSizeSnafu {
                sequences: packed.sequence_count(),
            }
            .fail();
        }
        let tokens = packed
            .sequence_length(0)
            .ok_or_else(|| PackedLayoutSnafu.build())?;
        let last_query = PagedDecodePlan::try_from_dimensions(
            packed
                .committed_offset(0)
                .ok_or_else(|| PackedLayoutSnafu.build())?
                .checked_add(tokens)
                .ok_or_else(|| {
                    PrefillDimensionOverflowSnafu {
                        dimensions: "committed offset + tokens",
                    }
                    .build()
                })?,
            query_heads,
            kv_heads,
            head_width,
        )
        .map_err(|source| PagedPrefillError::Decode { source })?;
        let query_elements = tokens
            .checked_mul(last_query.query_elements)
            .ok_or_else(|| {
                PrefillDimensionOverflowSnafu {
                    dimensions: "tokens * query_heads * head_width",
                }
                .build()
            })?;
        std::alloc::Layout::array::<f32>(query_elements).context(PrefillAllocationLayoutSnafu {
            allocation: "prefill query/output span",
            elements: query_elements,
        })?;
        Ok(Self {
            last_query,
            tokens,
            query_elements,
        })
    }

    /// Return the committed prefix length preceding this chunk.
    #[must_use]
    pub fn offset(self) -> usize {
        self.last_query.visible_tokens() - self.tokens
    }

    /// Return the active token count in this chunk.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Return the final causal visible-prefix length for the chunk's last row.
    #[must_use]
    pub const fn visible_tokens(self) -> usize {
        self.last_query.visible_tokens()
    }

    /// Return the admitted query-head count.
    #[must_use]
    pub const fn query_heads(self) -> usize {
        self.last_query.query_heads()
    }

    /// Return the admitted key/value-head count.
    #[must_use]
    pub const fn kv_heads(self) -> usize {
        self.last_query.kv_heads()
    }

    /// Return the admitted key/value head width.
    #[must_use]
    pub const fn head_width(self) -> usize {
        self.last_query.head_width()
    }

    /// Return the contiguous query-heads-per-key/value-head group.
    #[must_use]
    pub const fn gqa_group(self) -> usize {
        self.last_query.gqa_group()
    }

    /// Return the decoder-compatible f32 attention scale.
    #[must_use]
    pub const fn scale(self) -> f32 {
        self.last_query.scale()
    }

    /// Return the active query extent `[tokens, query_heads, head_width]`.
    #[must_use]
    pub const fn query_elements(self) -> usize {
        self.query_elements
    }

    /// Return the active output extent `[tokens, query_heads, head_width]`.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.query_elements
    }

    /// Return query `token`'s exact causal visible prefix.
    ///
    /// # Errors
    ///
    /// Returns [`PagedPrefillError::TokenOutOfRange`] when `token` is not an
    /// active row in this chunk.
    pub fn visible_tokens_for(self, token: usize) -> PagedPrefillResult<usize> {
        if token >= self.tokens {
            return PrefillTokenOutOfRangeSnafu {
                token,
                tokens: self.tokens,
            }
            .fail();
        }
        self.offset()
            .checked_add(token)
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| {
                PrefillDimensionOverflowSnafu {
                    dimensions: "committed offset + query token + one",
                }
                .build()
            })
    }
}

/// Failures while admitting or evaluating B=1 causal paged prefill.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum PagedPrefillError {
    /// The borrowed packed operation described something other than B=1.
    #[snafu(display("{PAGED_DECODE}: paged prefill requires B=1, got {sequences} sequences"))]
    BatchSize {
        /// Number of sequences in the packed operation.
        sequences: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The packed operation contradicted its checked B=1 accessors.
    #[snafu(display("{PAGED_DECODE}: packed B=1 plan has no sequence zero"))]
    PackedLayout {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A required prefill span arithmetic operation overflowed.
    #[snafu(
        display("{PAGED_DECODE}: prefill {dimensions} overflows usize"),
        context(name(PrefillDimensionOverflowSnafu))
    )]
    DimensionOverflow {
        /// The failed dimension calculation.
        dimensions: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The active query/output span cannot form a Rust layout.
    #[snafu(
        display("{PAGED_DECODE}: {allocation} layout for {elements} elements is unrepresentable"),
        context(name(PrefillAllocationLayoutSnafu))
    )]
    AllocationLayout {
        /// Span role.
        allocation: &'static str,
        /// Exact element count.
        elements: usize,
        /// Layout failure reported by the standard library.
        source: std::alloc::LayoutError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A requested chunk-relative query token was outside the active chunk.
    #[snafu(display("{PAGED_DECODE}: prefill token {token} is outside 0..{tokens}"))]
    TokenOutOfRange {
        /// Requested chunk-relative token.
        token: usize,
        /// Active chunk token count.
        tokens: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The canonical Q=1 operation rejected a shared attention condition.
    #[snafu(transparent)]
    Decode {
        /// Source Q=1 error.
        source: PagedDecodeError,
    },

    /// The output allocation could not reserve its checked active extent.
    #[snafu(
        display("{PAGED_DECODE}: could not reserve {elements} f32 elements for prefill output"),
        context(name(PrefillAllocationSnafu))
    )]
    Allocation {
        /// Exact requested output extent.
        elements: usize,
        /// Allocation failure reported by the standard library.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The query backing did not cover the active chunk extent.
    #[snafu(display(
        "{PAGED_DECODE}: prefill query length {actual} does not match expected {expected}"
    ))]
    QueryLengthMismatch {
        /// Required active query extent.
        expected: usize,
        /// Supplied query backing extent.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// A typed failure from B=1 prefill or one caller-owned row borrow.
#[derive(Debug, Snafu)]
#[snafu(module, visibility(pub))]
#[non_exhaustive]
pub enum PagedPrefillRowsError<E: std::error::Error + 'static> {
    /// The prefill operation rejected its geometry, input, or arithmetic.
    #[snafu(display("{PAGED_DECODE}: prefill operation failure: {source}"))]
    Kernel {
        /// Source operation failure.
        source: PagedPrefillError,
    },

    /// The caller could not borrow a key row at one absolute cache token.
    #[snafu(display("{PAGED_DECODE}: prefill key row {token} unavailable: {source}"))]
    KeyRow {
        /// Absolute cache token.
        token: usize,
        /// Typed caller-owned row-borrow failure.
        source: E,
    },

    /// The caller could not borrow a value row at one absolute cache token.
    #[snafu(display("{PAGED_DECODE}: prefill value row {token} unavailable: {source}"))]
    ValueRow {
        /// Absolute cache token.
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
    E: std::error::Error + 'static,
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

/// Evaluate every active B=1 query row against its exact causal cache prefix.
///
/// `queries` is the active `[tokens, query_heads, head_width]` prefix. The
/// row providers use absolute cache-token indices, so token `t` only visits
/// `0..plan.offset() + t + 1`. Each query-head evaluation delegates to the
/// established Q=1 implementation, preserving its materialized-score f32
/// order and grouped-query head selection.
///
/// # Errors
///
/// Returns [`PagedPrefillRowsError::Kernel`] for active-query shape,
/// allocation, Q=1 geometry, input, or arithmetic failures. A provider's
/// typed failure remains the source of [`PagedPrefillRowsError::KeyRow`] or
/// [`PagedPrefillRowsError::ValueRow`].
pub fn paged_prefill_cpu<'rows, E, KeyRow, ValueRow>(
    plan: PagedPrefillPlan,
    queries: &[f32],
    mut key_row: KeyRow,
    mut value_row: ValueRow,
) -> PagedPrefillRowsResult<Vec<f32>, E>
where
    E: std::error::Error + 'static,
    KeyRow: FnMut(usize) -> core::result::Result<&'rows [f32], E>,
    ValueRow: FnMut(usize) -> core::result::Result<&'rows [f32], E>,
{
    if queries.len() != plan.query_elements() {
        return Err(PagedPrefillRowsError::Kernel {
            source: QueryLengthMismatchSnafu {
                expected: plan.query_elements(),
                actual: queries.len(),
            }
            .build(),
        });
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(plan.output_elements())
        .context(PrefillAllocationSnafu {
            elements: plan.output_elements(),
        })
        .map_err(|source| PagedPrefillRowsError::Kernel { source })?;
    let query_head_elements = plan
        .query_heads()
        .checked_mul(plan.head_width())
        .ok_or_else(|| PagedPrefillRowsError::Kernel {
            source: PrefillDimensionOverflowSnafu {
                dimensions: "query_heads * head_width",
            }
            .build(),
        })?;
    for token in 0..plan.tokens() {
        let visible_tokens = plan
            .visible_tokens_for(token)
            .map_err(|source| PagedPrefillRowsError::Kernel { source })?;
        let query_plan = PagedDecodePlan::try_from_dimensions(
            visible_tokens,
            plan.query_heads(),
            plan.kv_heads(),
            plan.head_width(),
        )
        .map_err(|source| PagedPrefillRowsError::Kernel {
            source: PagedPrefillError::Decode { source },
        })?;
        let token_start = token.checked_mul(query_head_elements).ok_or_else(|| {
            PagedPrefillRowsError::Kernel {
                source: PrefillDimensionOverflowSnafu {
                    dimensions: "token * query_heads * head_width",
                }
                .build(),
            }
        })?;
        for query_head in 0..plan.query_heads() {
            let query_start = query_head
                .checked_mul(plan.head_width())
                .and_then(|head_start| token_start.checked_add(head_start))
                .ok_or_else(|| PagedPrefillRowsError::Kernel {
                    source: PrefillDimensionOverflowSnafu {
                        dimensions: "prefill query row offset",
                    }
                    .build(),
                })?;
            let query_end = query_start.checked_add(plan.head_width()).ok_or_else(|| {
                PagedPrefillRowsError::Kernel {
                    source: PrefillDimensionOverflowSnafu {
                        dimensions: "prefill query row end",
                    }
                    .build(),
                }
            })?;
            let head_output = paged_decode_cpu(
                query_plan,
                query_head,
                &queries[query_start..query_end],
                &mut key_row,
                &mut value_row,
            )
            .map_err(|error| match error {
                PagedDecodeRowsError::Kernel { source } => PagedPrefillRowsError::Kernel {
                    source: PagedPrefillError::Decode { source },
                },
                PagedDecodeRowsError::KeyRow { token, source } => {
                    PagedPrefillRowsError::KeyRow { token, source }
                }
                PagedDecodeRowsError::ValueRow { token, source } => {
                    PagedPrefillRowsError::ValueRow { token, source }
                }
            })?;
            output.extend_from_slice(&head_output);
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
    // SAFETY: this raw boundary retains its documented caller-owned device,
    // aliasing, table, and numerical-domain obligations.
    unsafe {
        launch_paged_decode_q1_f32_with_status(
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
            core::ptr::null_mut(),
        )
    }
}

/// Launch checked all-query-head native Q=1 paged attention.
///
/// This shares the raw launcher's exact descriptor and span validation and
/// numerical order. The session-owned status is read only after the same stream
/// synchronizes; a sticky fault must prevent logical publication.
///
/// # Errors
///
/// Returns the raw launcher's validation, stream-current, and launch failures.
/// The caller must separately read `status` after successful synchronization to
/// surface any classified native numerical-domain violation.
///
/// # Safety
///
/// Each nonempty pointer must remain live on `stream`'s device through stream
/// completion, and output must be exclusive and non-aliasing. The caller still
/// guarantees the selected device's qualified compiler and floating-point
/// profile, table bounds, and stream ordering. Checked classification removes
/// the raw path's caller obligation to prove all explicit arithmetic operands
/// and intermediates are finite normal-or-zero; it does not qualify math-library
/// internals or physical device denorm behavior.
#[cfg(feature = "gpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "the checked path preserves the raw Q=1 ABI and appends its borrowed status owner"
)]
pub unsafe fn launch_paged_decode_q1_f32_checked(
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
    status: &NativeNumericalStatus,
) -> KernelResult<()> {
    // SAFETY: the checked boundary retains raw pointer lifetime, aliasing, table,
    // device-profile, and stream-ordering obligations; status remains owned by its session.
    unsafe {
        launch_paged_decode_q1_f32_with_status(
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
            status.as_device_ptr(),
        )
    }
}

#[cfg(feature = "gpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "one shared implementation owns the fixed raw Q=1 ABI plus its nullable status pointer"
)]
unsafe fn launch_paged_decode_q1_f32_with_status(
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
    numerical_status: *mut u32,
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
            numerical_status,
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
        // SAFETY: both paths uphold device ownership, lifetime, concurrent
        // access, table-value, and stream-ordering obligations. The raw path
        // additionally upholds its numerical-domain obligation; the checked
        // path retains its status owner to same-stream synchronization.
        // Descriptor and span validation establish ABI extents.
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
                numerical_status.cast::<c_void>(),
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
    use crate::numerical_status::{NativeNumericalStatusCategory, NativeNumericalStatusMask};

    #[test]
    fn checked_native_first_token_maximum_sentinel_is_not_an_operand_failure()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let score = 1.0_f32;
        let maximum = f32::NEG_INFINITY.max(score);
        let bits = arithmetic_status_bits(maximum);
        let mask = NativeNumericalStatusMask::from_bits(bits)
            .ok_or("independent normal sentinel witness must use known status bits")?;

        assert!(mask.is_empty());
        Ok(())
    }

    #[test]
    fn checked_native_tree_ignores_dead_lane_overflow() {
        let contribution = f32::MAX * 0.75_f32;
        let mut lanes = [0.0_f32; 32];
        lanes[31] = contribution;

        for offset in [16_usize, 8, 4, 2, 1] {
            let prior = lanes;
            for lane in 0..offset {
                lanes[lane] += prior[lane + offset];
            }
        }
        let score = lanes[0] * (1.0_f32 / 32.0_f32.sqrt());
        let dead_lane_sum = contribution + contribution;

        assert!(contribution.is_normal());
        assert!(score.is_finite());
        assert!(dead_lane_sum.is_infinite());
        assert!(
            arithmetic_status_bits(score) == 0,
            "the live fixed-tree score is legitimate and must not inherit a dead lane's overflow"
        );
        assert_eq!(
            arithmetic_status_bits(dead_lane_sum),
            NativeNumericalStatusCategory::ArithmeticNonFinite.bit(),
            "the unguarded dead-lane addition would have set a sticky fault"
        );
    }

    #[test]
    fn checked_native_status_keeps_a_hidden_subnormal_product_sticky()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let left = 1.0e-20_f32;
        let right = 1.0e-20_f32;
        assert!(left.is_normal());
        assert!(right.is_normal());

        let product = left * right;
        let restored = product / left;
        assert!(restored.is_finite());
        assert!(restored.is_normal());

        let bits = arithmetic_status_bits(product) | arithmetic_status_bits(restored);
        let mask = NativeNumericalStatusMask::from_bits(bits)
            .ok_or("independent hidden-intermediate witness must use known status bits")?;

        assert!(mask.contains(NativeNumericalStatusCategory::ArithmeticSubnormal));
        Ok(())
    }

    fn arithmetic_status_bits(value: f32) -> u32 {
        const EXPONENT_MASK: u32 = 0x7f80_0000;
        const SIGNIFICAND_MASK: u32 = 0x007f_ffff;
        let bits = value.to_bits();
        if bits & EXPONENT_MASK == 0 && bits & SIGNIFICAND_MASK != 0 {
            return NativeNumericalStatusCategory::ArithmeticSubnormal.bit();
        }
        if bits & EXPONENT_MASK == EXPONENT_MASK {
            return NativeNumericalStatusCategory::ArithmeticNonFinite.bit();
        }
        0
    }

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
    fn prefill_cpu_matches_independent_f64_causal_gqa_oracle()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        const TOKENS: usize = 3;
        const OFFSET: usize = 2;
        const QUERY_HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const HEAD_WIDTH: usize = 3;

        let packed = PackedPrefillPlan::new(&[TOKENS], &[OFFSET], OFFSET + TOKENS)?;
        let plan =
            PagedPrefillPlan::try_from_packed_prefill(&packed, QUERY_HEADS, KV_HEADS, HEAD_WIDTH)?;
        let queries = prefill_queries(TOKENS, QUERY_HEADS, HEAD_WIDTH);
        let keys = prefill_rows(OFFSET + TOKENS, KV_HEADS, HEAD_WIDTH, 0.25);
        let values = prefill_rows(OFFSET + TOKENS, KV_HEADS, HEAD_WIDTH, 1.75);
        let actual = paged_prefill_cpu(
            plan,
            &queries,
            |token| {
                Ok::<_, std::convert::Infallible>(
                    &keys[token * KV_HEADS * HEAD_WIDTH..(token + 1) * KV_HEADS * HEAD_WIDTH],
                )
            },
            |token| {
                Ok::<_, std::convert::Infallible>(
                    &values[token * KV_HEADS * HEAD_WIDTH..(token + 1) * KV_HEADS * HEAD_WIDTH],
                )
            },
        )?;
        let oracle = prefill_oracle_f64(
            plan,
            &queries,
            &keys,
            &values,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_WIDTH,
        )?;
        assert_eq!(actual.len(), oracle.len());
        for (actual, oracle) in actual.iter().zip(oracle) {
            assert_relative_eq!(*actual as f64, oracle, epsilon = 1.0e-5_f64);
        }

        let mut future_mutated = values.clone();
        let future_start = (OFFSET + TOKENS - 1) * KV_HEADS * HEAD_WIDTH;
        for value in &mut future_mutated[future_start..] {
            *value += 1_000.0;
        }
        let future_actual = paged_prefill_cpu(
            plan,
            &queries,
            |token| {
                Ok::<_, std::convert::Infallible>(
                    &keys[token * KV_HEADS * HEAD_WIDTH..(token + 1) * KV_HEADS * HEAD_WIDTH],
                )
            },
            |token| {
                Ok::<_, std::convert::Infallible>(
                    &future_mutated
                        [token * KV_HEADS * HEAD_WIDTH..(token + 1) * KV_HEADS * HEAD_WIDTH],
                )
            },
        )?;
        let immutable_prefix = (TOKENS - 1) * QUERY_HEADS * HEAD_WIDTH;
        assert_eq!(
            &actual[..immutable_prefix],
            &future_actual[..immutable_prefix],
            "a staged future value row must not reach earlier causal queries"
        );
        assert_ne!(
            &actual[immutable_prefix..],
            &future_actual[immutable_prefix..],
            "the terminal query must retain visibility of its own staged row"
        );
        Ok(())
    }

    #[test]
    fn prefill_cpu_short_chunk_continues_from_the_prior_prefix()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        const QUERY_HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_WIDTH: usize = 2;

        let keys = prefill_rows(4, KV_HEADS, HEAD_WIDTH, 0.5);
        let values = prefill_rows(4, KV_HEADS, HEAD_WIDTH, 2.0);
        let packed = PackedPrefillPlan::new(&[1], &[3], 4)?;
        let plan =
            PagedPrefillPlan::try_from_packed_prefill(&packed, QUERY_HEADS, KV_HEADS, HEAD_WIDTH)?;
        assert_eq!(plan.offset(), 3);
        assert_eq!(plan.visible_tokens_for(0)?, 4);
        let queries = prefill_queries(1, QUERY_HEADS, HEAD_WIDTH);
        let actual = paged_prefill_cpu(
            plan,
            &queries,
            |token| {
                Ok::<_, std::convert::Infallible>(
                    &keys[token * KV_HEADS * HEAD_WIDTH..(token + 1) * KV_HEADS * HEAD_WIDTH],
                )
            },
            |token| {
                Ok::<_, std::convert::Infallible>(
                    &values[token * KV_HEADS * HEAD_WIDTH..(token + 1) * KV_HEADS * HEAD_WIDTH],
                )
            },
        )?;
        let oracle = prefill_oracle_f64(
            plan,
            &queries,
            &keys,
            &values,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_WIDTH,
        )?;
        for (actual, oracle) in actual.iter().zip(oracle) {
            assert_relative_eq!(*actual as f64, oracle, epsilon = 1.0e-5_f64);
        }
        Ok(())
    }

    #[test]
    fn prefill_plan_refuses_batch_and_overflowing_attention_geometry()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let batched = PackedPrefillPlan::new(&[1, 1], &[0, 0], 1)?;
        assert!(matches!(
            PagedPrefillPlan::try_from_packed_prefill(&batched, 1, 1, 1),
            Err(PagedPrefillError::BatchSize { sequences: 2, .. })
        ));

        let packed = PackedPrefillPlan::new(&[1], &[0], 1)?;
        assert!(matches!(
            PagedPrefillPlan::try_from_packed_prefill(&packed, usize::MAX, 1, 2),
            Err(PagedPrefillError::Decode {
                source: PagedDecodeError::DimensionOverflow { .. }
            })
        ));
        Ok(())
    }

    fn prefill_queries(tokens: usize, query_heads: usize, head_width: usize) -> Vec<f32> {
        (0..tokens * query_heads * head_width)
            .map(|index| 0.1 + index as f32 * 0.03)
            .collect()
    }

    fn prefill_rows(tokens: usize, kv_heads: usize, head_width: usize, bias: f32) -> Vec<f32> {
        (0..tokens * kv_heads * head_width)
            .map(|index| bias + index as f32 * 0.07)
            .collect()
    }

    fn prefill_oracle_f64(
        plan: PagedPrefillPlan,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        query_heads: usize,
        kv_heads: usize,
        head_width: usize,
    ) -> PagedPrefillResult<Vec<f64>> {
        let gqa_group = query_heads / kv_heads;
        let scale = 1.0_f64 / (head_width as f64).sqrt();
        let mut output = Vec::new();
        for token in 0..plan.tokens() {
            let visible_tokens = plan.visible_tokens_for(token)?;
            for query_head in 0..query_heads {
                let query_start = (token * query_heads + query_head) * head_width;
                let query = &queries[query_start..query_start + head_width];
                let kv_head = query_head / gqa_group;
                let scores = (0..visible_tokens)
                    .map(|key_token| {
                        let key_start = (key_token * kv_heads + kv_head) * head_width;
                        query
                            .iter()
                            .zip(&keys[key_start..key_start + head_width])
                            .map(|(query, key)| *query as f64 * *key as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let normalizer = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .sum::<f64>();
                for column in 0..head_width {
                    let weighted = scores
                        .iter()
                        .enumerate()
                        .map(|(key_token, score)| {
                            let value_index =
                                (key_token * kv_heads + kv_head) * head_width + column;
                            (score - maximum).exp() / normalizer * values[value_index] as f64
                        })
                        .sum::<f64>();
                    output.push(weighted);
                }
            }
        }
        Ok(output)
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
        let online = f64_online_oracle(&query, 0, &keys, &values);
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
    ) -> Vec<f64> {
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
        output
    }
}
