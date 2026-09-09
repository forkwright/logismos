//! Checked single-query paged-attention primitives.
//!
//! The CPU operation accepts logical key/value rows through fallible borrows;
//! it neither owns cache pages nor interprets a cache table. The optional HIP
//! launcher has a separate, explicit dense physical-page descriptor.
use num_traits::ToPrimitive;
use snafu::{ResultExt, Snafu};

const PAGED_DECODE: &str = "paged_decode";
/// Result alias for the logical paged-decode operation.
pub type PagedDecodeResult<T> = core::result::Result<T, PagedDecodeError>;

/// Result alias for a logical paged-decode operation with caller-owned rows.
pub type PagedDecodeRowsResult<T, E> = core::result::Result<T, PagedDecodeRowsError<E>>;

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
    scale: f32,
    workspace: PagedDecodeWorkspace,
}

impl PagedDecodePlan {
    /// Admit one Q=1 causal visible prefix with contiguous grouped-query heads.
    ///
    /// # Errors
    ///
    /// Returns [`PagedDecodeError`] when a dimension is zero, query heads do
    /// not divide evenly across key/value heads, the f32 scale cannot be
    /// represented finitely, or the required workspace sum overflows.
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
        Ok(Self {
            visible_tokens,
            query_heads,
            kv_heads,
            head_width,
            gqa_group: query_heads / kv_heads,
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

    fn row_elements(self) -> PagedDecodeResult<usize> {
        self.kv_heads.checked_mul(self.head_width).ok_or_else(|| {
            DimensionOverflowSnafu {
                dimensions: "kv_heads * head_width",
            }
            .build()
        })
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

    /// The head width could not yield a finite f32 attention scale.
    #[snafu(display("{PAGED_DECODE}: head_width {head_width} has no finite f32 scale"))]
    ScaleNotFinite {
        /// The rejected head width.
        head_width: usize,
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
pub enum PagedDecodeRowsError<E> {
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
/// normalizer, then output values all progress in ascending token order.
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
    let row_elements = plan
        .row_elements()
        .map_err(|source| PagedDecodeRowsError::Kernel { source })?;
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
        validate_finite("key row", row)
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
                index: 0,
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
        validate_finite("value row", row)
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
        assert_relative_eq!(output[0], 7.0, epsilon = 1.0e-5);
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
    fn independent_online_f64_witness_handles_repeated_maximums_and_page_boundary()
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

        for (materialized, online) in materialized.iter().zip(online) {
            assert_relative_eq!(*materialized, online, epsilon = 1.0e-12);
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
