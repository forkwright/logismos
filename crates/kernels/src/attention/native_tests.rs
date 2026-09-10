use approx::relative_eq;
use num_traits::ToPrimitive;
use snafu::Snafu;

use super::*;

const QUERY_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const PHYSICAL_PAGES: usize = 3;
const WAVE_LANES: usize = 32;
const REDUCTION_OFFSETS: [usize; 5] = [16, 8, 4, 2, 1];
const WELL_CONDITIONED_EPSILON: f32 = 1.0e-3;
const UNUSED_PHYSICAL_KEY: f32 = -777.0;
const UNUSED_PHYSICAL_VALUE: f32 = 333.0;

#[derive(Debug, Snafu)]
enum NativeReferenceError {
    #[snafu(display("native reference {input} extent {actual} does not match {expected}"))]
    Extent {
        input: &'static str,
        expected: usize,
        actual: usize,
    },

    #[snafu(display(
        "native reference page table entry {logical_page} selects physical page {physical_page}, outside 0..{physical_pages}"
    ))]
    PageTableEntry {
        logical_page: usize,
        physical_page: u32,
        physical_pages: usize,
    },

    #[snafu(display("native reference {input}[{index}] is not finite"))]
    NonFiniteInput { input: &'static str, index: usize },

    #[snafu(display("native reference {input}[{index}] is neither zero nor normal"))]
    NonNormalInput { input: &'static str, index: usize },

    #[snafu(display("native reference produced a non-finite {stage} at {index}"))]
    NonFiniteArithmetic { stage: &'static str, index: usize },

    #[snafu(display("native reference produced a non-normal {stage} at {index}"))]
    NonNormalArithmetic { stage: &'static str, index: usize },

    #[snafu(display("native reference {operation} overflowed usize"))]
    ArithmeticOverflow { operation: &'static str },

    #[snafu(display("native reference could not reserve {elements} elements for {allocation}"))]
    Allocation {
        allocation: &'static str,
        elements: usize,
        source: std::collections::TryReserveError,
    },

    #[snafu(display("native reference could not represent {value} as f32 for {input}"))]
    F32Conversion { input: &'static str, value: usize },

    #[snafu(display("native reference {input}[{index}] cannot be narrowed to f32"))]
    F32Narrowing { input: &'static str, index: usize },

    #[snafu(display("native reference could not represent {value} as f64 for {input}"))]
    F64Conversion { input: &'static str, value: usize },
}

#[derive(Debug)]
struct NativeFixture {
    native: NativePagedDecodePlan,
    query: Vec<f32>,
    logical_keys: Vec<Vec<f32>>,
    logical_values: Vec<Vec<f32>>,
    keys: Vec<f32>,
    values: Vec<f32>,
    table: Vec<u32>,
}

#[derive(Clone, Copy)]
struct CausalPrefillFixtureGeometry {
    offset: usize,
    tokens: usize,
    query_heads: usize,
    head_width: usize,
}

impl CausalPrefillFixtureGeometry {
    fn final_visible_tokens(self) -> Result<usize, NativeReferenceError> {
        self.offset
            .checked_add(self.tokens)
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "causal prefill fixture final visible tokens",
            })
    }

    fn visible_tokens_for(self, token: usize) -> Result<usize, NativeReferenceError> {
        if token >= self.tokens {
            return Err(NativeReferenceError::Extent {
                input: "causal prefill fixture token",
                expected: self.tokens,
                actual: token
                    .checked_add(1)
                    .ok_or(NativeReferenceError::ArithmeticOverflow {
                        operation: "causal prefill fixture token extent",
                    })?,
            });
        }
        self.offset
            .checked_add(token)
            .and_then(|position| position.checked_add(1))
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "causal prefill fixture visible tokens",
            })
    }
}

#[derive(Clone, Copy)]
enum DotReduction {
    NativeTree,
    Sequential,
}

#[derive(Clone, Copy)]
struct NativeReferenceMode {
    dot_reduction: DotReduction,
    rescale_prior_output: bool,
    include_tail: bool,
}

impl NativeReferenceMode {
    const NATIVE: Self = Self {
        dot_reduction: DotReduction::NativeTree,
        rescale_prior_output: true,
        include_tail: true,
    };

    const WITHOUT_RESCALE: Self = Self {
        dot_reduction: DotReduction::NativeTree,
        rescale_prior_output: false,
        include_tail: true,
    };

    const SEQUENTIAL_DOT: Self = Self {
        dot_reduction: DotReduction::Sequential,
        rescale_prior_output: true,
        include_tail: true,
    };

    const WITHOUT_TAIL: Self = Self {
        dot_reduction: DotReduction::NativeTree,
        rescale_prior_output: true,
        include_tail: false,
    };
}

#[test]
fn native_f32_matches_independent_logical_oracle_for_all_page_sizes_and_widths()
-> Result<(), NativeReferenceError> {
    for page_tokens in [8_usize, 16, 32] {
        for head_width in [7_usize, 37] {
            let fixture = qualified_fixture(page_tokens, head_width, page_tokens + 3)?;
            let actual = native_f32_reference(
                &fixture.native,
                &fixture.query,
                &fixture.keys,
                &fixture.values,
                &fixture.table,
            )?;
            let expected = f64_logical_materialized_oracle(
                &fixture,
                fixture.native.logical().visible_tokens(),
                false,
            )?;

            assert_close_to_f64(
                &actual,
                &expected,
                "native reference must retain well-conditioned logical attention",
            )?;

            let wrong_query = query_reusing_head_zero(&fixture)?;
            let wrong_query_head = native_f32_reference(
                &fixture.native,
                &wrong_query,
                &fixture.keys,
                &fixture.values,
                &fixture.table,
            )?;
            assert_f32_separated(
                &actual,
                &wrong_query_head,
                "query heads in one GQA group must retain their own query rows",
            )?;

            let mut wrong_table = fixture.table.clone();
            swap_first_two(&mut wrong_table)?;
            let wrong_page = native_f32_reference(
                &fixture.native,
                &fixture.query,
                &fixture.keys,
                &fixture.values,
                &wrong_table,
            )?;
            assert_f32_separated(
                &actual,
                &wrong_page,
                "page-table indirection must affect native rows",
            )?;

            let wrong_gqa = f64_logical_materialized_oracle(
                &fixture,
                fixture.native.logical().visible_tokens(),
                true,
            )?;
            assert_f64_separated(
                &expected,
                &wrong_gqa,
                "contiguous GQA selection must affect logical output",
            )?;

            let wrong_rescale = native_f32_reference_with_mode(
                &fixture.native,
                &fixture.query,
                &fixture.keys,
                &fixture.values,
                &fixture.table,
                NativeReferenceMode::WITHOUT_RESCALE,
            )?;
            assert_f32_separated(
                &actual,
                &wrong_rescale,
                "online maximum changes must rescale prior output",
            )?;
            if head_width > WAVE_LANES {
                let wrong_tail = native_f32_reference_with_mode(
                    &fixture.native,
                    &fixture.query,
                    &fixture.keys,
                    &fixture.values,
                    &fixture.table,
                    NativeReferenceMode::WITHOUT_TAIL,
                )?;
                assert_f32_separated(
                    &actual,
                    &wrong_tail,
                    "lane-strided dot accumulation must include coordinates past lane 31",
                )?;
                let tail_index =
                    head_width
                        .checked_sub(1)
                        .ok_or(NativeReferenceError::ArithmeticOverflow {
                            operation: "tail output index",
                        })?;
                let tail = actual.get(tail_index).ok_or(NativeReferenceError::Extent {
                    input: "native tail output",
                    expected: actual.len(),
                    actual: tail_index,
                })?;
                assert!(
                    tail.abs() > WELL_CONDITIONED_EPSILON,
                    "the final width tail coordinate must be materially populated"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn native_f32_bounds_the_visible_prefix_across_nonidentity_partial_pages()
-> Result<(), NativeReferenceError> {
    for page_tokens in [8_usize, 16, 32] {
        let visible_tokens = page_tokens + 3;
        let fixture = qualified_fixture(page_tokens, 7, visible_tokens)?;
        let prefix_tokens = page_tokens + 1;
        let prefix_logical = PagedDecodePlan::try_from_dimensions(
            prefix_tokens,
            QUERY_HEADS,
            KV_HEADS,
            fixture.native.logical().head_width(),
        )
        .map_err(|_| NativeReferenceError::ArithmeticOverflow {
            operation: "prefix logical plan",
        })?;
        let prefix_native = NativePagedDecodePlan::try_from_paged_decode(
            prefix_logical,
            page_tokens,
            fixture.native.physical_pages(),
        )
        .map_err(|_| NativeReferenceError::ArithmeticOverflow {
            operation: "prefix native plan",
        })?;
        let prefix_actual = native_f32_reference(
            &prefix_native,
            &fixture.query,
            &fixture.keys,
            &fixture.values,
            &fixture.table,
        )?;
        let prefix_expected = f64_logical_materialized_oracle(&fixture, prefix_tokens, false)?;
        assert_close_to_f64(
            &prefix_actual,
            &prefix_expected,
            "native descriptor must stop at the admitted causal prefix",
        )?;

        let extended_actual = native_f32_reference(
            &fixture.native,
            &fixture.query,
            &fixture.keys,
            &fixture.values,
            &fixture.table,
        )?;
        assert_f32_separated(
            &prefix_actual,
            &extended_actual,
            "rows past the admitted prefix must not participate in native attention",
        )?;
    }
    Ok(())
}

#[test]
fn native_f32_lane_tree_has_a_distinct_cancellation_sensitive_witness()
-> Result<(), NativeReferenceError> {
    let fixture = cancellation_fixture()?;
    let native = native_f32_reference(
        &fixture.native,
        &fixture.query,
        &fixture.keys,
        &fixture.values,
        &fixture.table,
    )?;
    let sequential = native_f32_reference_with_mode(
        &fixture.native,
        &fixture.query,
        &fixture.keys,
        &fixture.values,
        &fixture.table,
        NativeReferenceMode::SEQUENTIAL_DOT,
    )?;

    assert_ne!(
        native, sequential,
        "the fixed 32-lane reduction tree must not silently become sequential accumulation"
    );
    Ok(())
}

#[test]
fn native_f32_reference_refuses_malformed_extents_table_and_nonfinite_arithmetic()
-> Result<(), NativeReferenceError> {
    let fixture = qualified_fixture(8, 7, 11)?;

    let short_query = fixture
        .query
        .get(
            ..fixture.query.len().checked_sub(1).ok_or(
                NativeReferenceError::ArithmeticOverflow {
                    operation: "short query length",
                },
            )?,
        )
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "short query slice",
        })?;
    assert!(
        matches!(
            native_f32_reference(
                &fixture.native,
                short_query,
                &fixture.keys,
                &fixture.values,
                &fixture.table,
            ),
            Err(NativeReferenceError::Extent { input: "query", .. })
        ),
        "short query must be rejected before native-order evaluation"
    );

    let mut invalid_table = fixture.table.clone();
    let first_entry =
        invalid_table
            .first_mut()
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "invalid table entry",
            })?;
    *first_entry = u32::try_from(fixture.native.physical_pages()).map_err(|_| {
        NativeReferenceError::ArithmeticOverflow {
            operation: "physical page to u32",
        }
    })?;
    assert!(
        matches!(
            native_f32_reference(
                &fixture.native,
                &fixture.query,
                &fixture.keys,
                &fixture.values,
                &invalid_table,
            ),
            Err(NativeReferenceError::PageTableEntry {
                logical_page: 0,
                ..
            })
        ),
        "out-of-range physical page must be rejected by the qualification reference"
    );

    let mut nonfinite_keys = fixture.keys.clone();
    let first_key = nonfinite_keys
        .first_mut()
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "nonfinite key",
        })?;
    *first_key = f32::NAN;
    assert!(
        matches!(
            native_f32_reference(
                &fixture.native,
                &fixture.query,
                &nonfinite_keys,
                &fixture.values,
                &fixture.table,
            ),
            Err(NativeReferenceError::NonFiniteInput { input: "keys", .. })
        ),
        "non-finite physical key data must be rejected before evaluation"
    );

    let selected_key_index = physical_row_start(fixture.native, &fixture.table, 0, 0)?;
    let mut subnormal_keys = fixture.keys.clone();
    let subnormal_key = subnormal_keys.get_mut(selected_key_index).ok_or(
        NativeReferenceError::ArithmeticOverflow {
            operation: "subnormal key",
        },
    )?;
    *subnormal_key = f32::from_bits(1);
    assert!(
        matches!(
            native_f32_reference(
                &fixture.native,
                &fixture.query,
                &subnormal_keys,
                &fixture.values,
                &fixture.table,
            ),
            Err(NativeReferenceError::NonNormalInput { input: "keys", .. })
        ),
        "subnormal physical input must violate the native zero-or-normal domain"
    );

    let mut subnormal_query = fixture.query.clone();
    let first_subnormal_query =
        subnormal_query
            .first_mut()
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "subnormal arithmetic query",
            })?;
    *first_subnormal_query = f32::MIN_POSITIVE;
    let mut normal_half_keys = fixture.keys.clone();
    let first_normal_half_key = normal_half_keys.get_mut(selected_key_index).ok_or(
        NativeReferenceError::ArithmeticOverflow {
            operation: "subnormal arithmetic key",
        },
    )?;
    *first_normal_half_key = 0.5;
    assert!(
        matches!(
            native_f32_reference(
                &fixture.native,
                &subnormal_query,
                &normal_half_keys,
                &fixture.values,
                &fixture.table,
            ),
            Err(NativeReferenceError::NonNormalArithmetic {
                stage: "dot product",
                ..
            })
        ),
        "normal inputs whose product is subnormal must violate the native arithmetic domain"
    );

    let mut overflowing_query = fixture.query.clone();
    let first_query =
        overflowing_query
            .first_mut()
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "overflowing query",
            })?;
    *first_query = f32::MAX;
    let mut overflowing_keys = fixture.keys.clone();
    let first_overflowing_key = overflowing_keys.get_mut(selected_key_index).ok_or(
        NativeReferenceError::ArithmeticOverflow {
            operation: "overflowing key",
        },
    )?;
    *first_overflowing_key = f32::MAX;
    assert!(
        matches!(
            native_f32_reference(
                &fixture.native,
                &overflowing_query,
                &overflowing_keys,
                &fixture.values,
                &fixture.table,
            ),
            Err(NativeReferenceError::NonFiniteArithmetic {
                stage: "dot product",
                ..
            })
        ),
        "non-finite native-order products must be rejected by the qualification reference"
    );
    Ok(())
}

#[test]
fn native_f32_reference_refuses_masked_online_subnormal_boundaries()
-> Result<(), NativeReferenceError> {
    let logical = PagedDecodePlan::try_from_dimensions(2, 1, 1, 1).map_err(|_| {
        NativeReferenceError::ArithmeticOverflow {
            operation: "online subnormal logical plan",
        }
    })?;
    let native = NativePagedDecodePlan::try_from_paged_decode(logical, 8, 1).map_err(|_| {
        NativeReferenceError::ArithmeticOverflow {
            operation: "online subnormal native plan",
        }
    })?;
    let query = [1.0_f32];
    let mut keys = vec![0.0_f32; native.key_value_elements()];
    let mut values = vec![0.0_f32; native.key_value_elements()];
    assign(&mut keys, 1, 80.0, "online subnormal key")?;
    assign(&mut values, 0, 1.0e-5, "online subnormal prior value")?;
    assign(&mut values, 1, 1.0, "online subnormal next value")?;
    assert!(
        matches!(
            native_f32_reference(&native, &query, &keys, &values, &[0_u32]),
            Err(NativeReferenceError::NonNormalArithmetic {
                stage: "prior output product",
                ..
            })
        ),
        "a subnormal rescaled prior output must not be masked by the next token output"
    );

    let first_score = f32::MIN_POSITIVE;
    let second_score = f32::from_bits(first_score.to_bits().checked_add(1).ok_or(
        NativeReferenceError::ArithmeticOverflow {
            operation: "adjacent normal score bits",
        },
    )?);
    let mut delta_keys = vec![0.0_f32; native.key_value_elements()];
    assign(
        &mut delta_keys,
        0,
        first_score,
        "first adjacent normal score",
    )?;
    assign(
        &mut delta_keys,
        1,
        second_score,
        "second adjacent normal score",
    )?;
    assert!(
        matches!(
            native_f32_reference(&native, &query, &delta_keys, &values, &[0_u32]),
            Err(NativeReferenceError::NonNormalArithmetic {
                stage: "maximum delta",
                ..
            })
        ),
        "a subnormal maximum subtraction must be rejected before exponentiation"
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
fn reserved_device_q1_paged_attention_matches_native_fixture()
-> Result<(), Box<dyn std::error::Error>> {
    use hipcore::{Device, DeviceBuffer, Stream};

    let fixture = qualified_fixture(8, 37, 11)?;
    let native_expected = native_f32_reference(
        &fixture.native,
        &fixture.query,
        &fixture.keys,
        &fixture.values,
        &fixture.table,
    )?;
    let expected = f64_logical_materialized_oracle(
        &fixture,
        fixture.native.logical().visible_tokens(),
        false,
    )?;
    let device = Device::new(0)?;
    let stream = Stream::new(&device)?;
    let query = DeviceBuffer::from_host(&device, &fixture.query)?;
    let keys = DeviceBuffer::from_host(&device, &fixture.keys)?;
    let values = DeviceBuffer::from_host(&device, &fixture.values)?;
    let table = DeviceBuffer::from_host(&device, &fixture.table)?;
    let initialized_output = vec![0.0_f32; fixture.native.output_elements()];
    let output = DeviceBuffer::from_host(&device, &initialized_output)?;
    let status = crate::numerical_status::NativeNumericalStatus::new(&device)?;
    // SAFETY: the fixture supplies distinct descriptor-sized buffers, keeps
    // them live through synchronization, and uses finite physical data with
    // in-range page-table entries.
    unsafe {
        launch_paged_decode_q1_f32_checked(
            fixture.native,
            query.as_device_ptr(),
            query.len(),
            keys.as_device_ptr(),
            keys.len(),
            values.as_device_ptr(),
            values.len(),
            table.as_device_ptr(),
            table.len(),
            output.as_device_ptr(),
            output.len(),
            &stream,
            &status,
        )?;
    }
    stream.synchronize()?;
    status.read_after_synchronization()?;
    let mut actual = vec![0.0_f32; output.len()];
    output.copy_to_host(&mut actual)?;
    for (index, value) in actual.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(NativeReferenceError::NonFiniteArithmetic {
                stage: "device output",
                index,
            }
            .into());
        }
    }
    assert_close_to_f32(
        &actual,
        &native_expected,
        "reserved device output must retain the native lane/tree order",
    )?;
    assert_close_to_f64(
        &actual,
        &expected,
        "reserved device output must match the independent logical fixture",
    )?;
    Ok(())
}

#[test]
#[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
fn reserved_device_prefill_causal_multiquery_matches_f64_future_leak_fixture()
-> Result<(), Box<dyn std::error::Error>> {
    use hipcore::{Device, DeviceBuffer, Stream};

    const TOKENS: usize = 2;
    const PREFILL_QUERY_HEADS: usize = 2;
    const PREFILL_KV_HEADS: usize = 1;
    const PREFILL_HEAD_WIDTH: usize = 1;
    const PREFILL_PHYSICAL_PAGES: usize = 3;

    let device = Device::new(0)?;
    let stream = Stream::new(&device)?;
    for page_tokens in [8_usize, 16, 32] {
        let geometry = CausalPrefillFixtureGeometry {
            offset: page_tokens - 1,
            tokens: TOKENS,
            query_heads: PREFILL_QUERY_HEADS,
            head_width: PREFILL_HEAD_WIDTH,
        };
        let final_visible_tokens = geometry.final_visible_tokens()?;
        let packed = crate::PackedPrefillPlan::new(
            &[geometry.tokens],
            &[geometry.offset],
            final_visible_tokens,
        )?;
        let logical = PagedPrefillPlan::try_from_packed_prefill(
            &packed,
            PREFILL_QUERY_HEADS,
            PREFILL_KV_HEADS,
            PREFILL_HEAD_WIDTH,
        )?;
        let native = NativePagedPrefillPlan::try_from_paged_prefill(
            logical,
            page_tokens,
            PREFILL_PHYSICAL_PAGES,
        )?;
        assert_eq!(
            native.logical().offset(),
            geometry.offset,
            "native B=1 prefill must retain the synthetic nonzero prefix"
        );
        assert_eq!(
            native.visible_tokens(),
            final_visible_tokens,
            "native B=1 prefill must retain the synthetic final visible prefix"
        );

        let fixture_values = causal_prefill_future_leak_values(geometry)?;
        let expected = causal_prefill_f64_oracle(geometry, &fixture_values)?;
        let mut table = reserve("causal prefill page table", native.page_table_entries())?;
        for logical_page in 0..native.page_table_entries() {
            table.push(u32::try_from(logical_page)?);
        }
        let keys_host = vec![0.0_f32; native.key_value_elements()];
        let mut values_host = vec![0.0_f32; native.key_value_elements()];
        for (token, value) in fixture_values.iter().copied().enumerate() {
            let logical_page = token / page_tokens;
            let in_page_token = token % page_tokens;
            let physical_page = usize::try_from(*table.get(logical_page).ok_or(
                NativeReferenceError::Extent {
                    input: "causal prefill page table",
                    expected: logical_page.checked_add(1).ok_or(
                        NativeReferenceError::ArithmeticOverflow {
                            operation: "causal prefill page table extent",
                        },
                    )?,
                    actual: table.len(),
                },
            )?)?;
            let physical_token = physical_page
                .checked_mul(page_tokens)
                .and_then(|page_start| page_start.checked_add(in_page_token))
                .ok_or(NativeReferenceError::ArithmeticOverflow {
                    operation: "causal prefill physical token",
                })?;
            // This fixture has one KV head of width one, so a physical token
            // is exactly one dense key/value element. The identity page table
            // still exercises the final row on a second page for B=8/16/32.
            assign(
                &mut values_host,
                physical_token,
                value,
                "causal prefill physical value",
            )?;
        }

        let query_host = vec![0.0_f32; native.query_elements()];
        let output_host = vec![0.0_f32; native.output_elements()];
        let query = DeviceBuffer::from_host(&device, &query_host)?;
        let keys = DeviceBuffer::from_host(&device, &keys_host)?;
        let values = DeviceBuffer::from_host(&device, &values_host)?;
        let page_table = DeviceBuffer::from_host(&device, &table)?;
        let output = DeviceBuffer::from_host(&device, &output_host)?;
        let status = crate::numerical_status::NativeNumericalStatus::new(&device)?;
        // SAFETY: each device buffer has the descriptor's admitted active
        // extent, the output is separate, and the synthetic page table maps
        // only initialized physical rows for the full stream lifetime.
        unsafe {
            launch_paged_prefill_b1_f32_checked(
                native,
                query.as_device_ptr(),
                query.len(),
                keys.as_device_ptr(),
                keys.len(),
                values.as_device_ptr(),
                values.len(),
                page_table.as_device_ptr(),
                page_table.len(),
                output.as_device_ptr(),
                output.len(),
                &stream,
                &status,
            )?;
        }
        stream.synchronize()?;
        status.read_after_synchronization()?;
        let mut actual = vec![0.0_f32; output.len()];
        output.copy_to_host(&mut actual)?;
        assert_close_to_f64(
            &actual,
            &expected,
            "causal multiquery device output must match the independent f64 fixture",
        )?;

        let first_expected = expected
            .first()
            .copied()
            .ok_or(NativeReferenceError::Extent {
                input: "causal prefill expected output",
                expected: 1,
                actual: expected.len(),
            })?;
        let first_actual = actual
            .first()
            .copied()
            .ok_or(NativeReferenceError::Extent {
                input: "causal prefill device output",
                expected: 1,
                actual: actual.len(),
            })?;
        assert!(
            relative_eq!(
                first_actual,
                first_expected
                    .to_f32()
                    .ok_or(NativeReferenceError::F32Narrowing {
                        input: "causal prefill first expected output",
                        index: 0,
                    })?,
                epsilon = WELL_CONDITIONED_EPSILON
            ),
            "the first causal row must retain its known prefix endpoint"
        );
        let future_leaking_mean = causal_prefill_f64_mean(&fixture_values, final_visible_tokens)?;
        assert!(
            !relative_eq!(
                first_actual,
                future_leaking_mean
                    .to_f32()
                    .ok_or(NativeReferenceError::F32Narrowing {
                        input: "causal prefill future-leaking output",
                        index: 0,
                    })?,
                epsilon = WELL_CONDITIONED_EPSILON
            ),
            "the first causal row must not read the staged future chunk row"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
fn reserved_device_q1_paged_attention_keeps_dead_lane_overflow_unpublished()
-> Result<(), Box<dyn std::error::Error>> {
    use hipcore::{Device, DeviceBuffer, Stream};

    let logical = PagedDecodePlan::try_from_dimensions(1, 1, 1, 32)?;
    let native = NativePagedDecodePlan::try_from_paged_decode(logical, 8, 1)?;
    let device = Device::new(0)?;
    let stream = Stream::new(&device)?;
    let mut query_host = vec![0.0_f32; native.query_elements()];
    query_host[31] = 1.0;
    let query = DeviceBuffer::from_host(&device, &query_host)?;
    let mut keys_host = vec![0.0_f32; native.key_value_elements()];
    keys_host[31] = f32::MAX * 0.75;
    let keys = DeviceBuffer::from_host(&device, &keys_host)?;
    let values_host = vec![0.0_f32; native.key_value_elements()];
    let values = DeviceBuffer::from_host(&device, &values_host)?;
    let table = DeviceBuffer::from_host(&device, &[0_u32])?;
    let output_host = vec![0.0_f32; native.output_elements()];
    let output = DeviceBuffer::from_host(&device, &output_host)?;
    let status = crate::numerical_status::NativeNumericalStatus::new(&device)?;

    // SAFETY: all allocations have exact descriptor extents, stay live through
    // synchronization, and page zero selects the initialized physical page.
    unsafe {
        launch_paged_decode_q1_f32_checked(
            native,
            query.as_device_ptr(),
            query.len(),
            keys.as_device_ptr(),
            keys.len(),
            values.as_device_ptr(),
            values.len(),
            table.as_device_ptr(),
            table.len(),
            output.as_device_ptr(),
            output.len(),
            &stream,
            &status,
        )?;
    }
    stream.synchronize()?;
    status.read_after_synchronization()?;
    let mut observed_output = vec![0.0_f32; output.len()];
    output.copy_to_host(&mut observed_output)?;
    assert!(observed_output.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
#[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
fn reserved_device_q1_paged_attention_rejects_hidden_subnormal_product()
-> Result<(), Box<dyn std::error::Error>> {
    use hipcore::{Device, DeviceBuffer, Stream};

    let logical = PagedDecodePlan::try_from_dimensions(1, 1, 1, 1)?;
    let native = NativePagedDecodePlan::try_from_paged_decode(logical, 8, 1)?;
    let device = Device::new(0)?;
    let stream = Stream::new(&device)?;
    let query = DeviceBuffer::from_host(&device, &[1.0e-20_f32])?;
    let mut keys_host = vec![0.0_f32; native.key_value_elements()];
    keys_host[0] = 1.0e-20_f32;
    let keys = DeviceBuffer::from_host(&device, &keys_host)?;
    let mut values_host = vec![0.0_f32; native.key_value_elements()];
    values_host[0] = 1.0_f32;
    let values = DeviceBuffer::from_host(&device, &values_host)?;
    let table = DeviceBuffer::from_host(&device, &[0_u32])?;
    let output = DeviceBuffer::from_host(&device, &[0.0_f32])?;
    let status = crate::numerical_status::NativeNumericalStatus::new(&device)?;

    // SAFETY: all device allocations have exact descriptor extents, remain live
    // through synchronization, and the single table entry selects page zero.
    unsafe {
        launch_paged_decode_q1_f32_checked(
            native,
            query.as_device_ptr(),
            query.len(),
            keys.as_device_ptr(),
            keys.len(),
            values.as_device_ptr(),
            values.len(),
            table.as_device_ptr(),
            table.len(),
            output.as_device_ptr(),
            output.len(),
            &stream,
            &status,
        )?;
    }
    stream.synchronize()?;
    match status.read_after_synchronization() {
        Err(crate::Error::NumericalStatus {
            source: crate::numerical_status::NativeNumericalStatusError::Observed { mask, .. },
            ..
        }) => {
            assert!(mask.contains(
                crate::numerical_status::NativeNumericalStatusCategory::ArithmeticSubnormal
            ));
        }
        Err(error) => return Err(format!("unexpected native status failure: {error}").into()),
        Ok(()) => {
            return Err("hidden subnormal product must reject checked native attention".into());
        }
    }
    Ok(())
}

fn qualified_fixture(
    page_tokens: usize,
    head_width: usize,
    visible_tokens: usize,
) -> Result<NativeFixture, NativeReferenceError> {
    let logical =
        PagedDecodePlan::try_from_dimensions(visible_tokens, QUERY_HEADS, KV_HEADS, head_width)
            .map_err(|_| NativeReferenceError::ArithmeticOverflow {
                operation: "qualified logical plan",
            })?;
    let native = NativePagedDecodePlan::try_from_paged_decode(logical, page_tokens, PHYSICAL_PAGES)
        .map_err(|_| NativeReferenceError::ArithmeticOverflow {
            operation: "qualified native plan",
        })?;
    let mut query = reserve("query", native.query_elements())?;
    for query_head in 0..QUERY_HEADS {
        for column in 0..head_width {
            query.push(fixture_query_component(query_head, column, head_width));
        }
    }

    let mut logical_keys = reserve("logical keys", visible_tokens)?;
    let mut logical_values = reserve("logical values", visible_tokens)?;
    for token in 0..visible_tokens {
        let mut key_row = reserve("logical key row", logical.row_elements())?;
        let mut value_row = reserve("logical value row", logical.row_elements())?;
        let token_value = checked_f32("logical token", token)?;
        for kv_head in 0..KV_HEADS {
            let head_marker = match kv_head {
                0 => 10.0,
                1 => 100.0,
                _ => {
                    return Err(NativeReferenceError::ArithmeticOverflow {
                        operation: "KV marker",
                    });
                }
            };
            for column in 0..head_width {
                let key = fixture_key_component(token, column, head_width);
                let column_value = checked_f32("logical column", column)?;
                key_row.push(key);
                value_row.push(head_marker + token_value + column_value * 0.01);
            }
        }
        logical_keys.push(key_row);
        logical_values.push(value_row);
    }

    let mut table = reserve("page table", native.page_table_entries())?;
    for logical_page in 0..native.page_table_entries() {
        table.push(nonidentity_physical_page(logical_page)?);
    }
    let mut keys = vec![UNUSED_PHYSICAL_KEY; native.key_value_elements()];
    let mut values = vec![UNUSED_PHYSICAL_VALUE; native.key_value_elements()];
    let key_elements = keys.len();
    let value_elements = values.len();
    for token in 0..visible_tokens {
        let row_start = fixture_physical_row_start(
            &table,
            token,
            page_tokens,
            logical.row_elements(),
            PHYSICAL_PAGES,
        )?;
        let row_end = row_start.checked_add(logical.row_elements()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "qualified physical row end",
            },
        )?;
        let key_destination =
            keys.get_mut(row_start..row_end)
                .ok_or(NativeReferenceError::Extent {
                    input: "keys",
                    expected: native.key_value_elements(),
                    actual: key_elements,
                })?;
        let value_destination =
            values
                .get_mut(row_start..row_end)
                .ok_or(NativeReferenceError::Extent {
                    input: "values",
                    expected: native.key_value_elements(),
                    actual: value_elements,
                })?;
        let logical_key =
            logical_keys
                .get(token)
                .ok_or(NativeReferenceError::ArithmeticOverflow {
                    operation: "logical key fixture row",
                })?;
        let logical_value =
            logical_values
                .get(token)
                .ok_or(NativeReferenceError::ArithmeticOverflow {
                    operation: "logical value fixture row",
                })?;
        key_destination.copy_from_slice(logical_key);
        value_destination.copy_from_slice(logical_value);
    }
    Ok(NativeFixture {
        native,
        query,
        logical_keys,
        logical_values,
        keys,
        values,
        table,
    })
}

fn cancellation_fixture() -> Result<NativeFixture, NativeReferenceError> {
    let logical = PagedDecodePlan::try_from_dimensions(2, 1, 1, 33).map_err(|_| {
        NativeReferenceError::ArithmeticOverflow {
            operation: "cancellation logical plan",
        }
    })?;
    let native = NativePagedDecodePlan::try_from_paged_decode(logical, 8, 1).map_err(|_| {
        NativeReferenceError::ArithmeticOverflow {
            operation: "cancellation native plan",
        }
    })?;
    let query = vec![1.0_f32; native.query_elements()];
    let mut logical_keys = reserve("cancellation logical keys", logical.visible_tokens())?;
    let mut logical_values = reserve("cancellation logical values", logical.visible_tokens())?;
    let mut first_key = vec![0.0_f32; logical.row_elements()];
    assign(&mut first_key, 0, 16_777_216.0, "first cancellation key")?;
    assign(&mut first_key, 1, -16_777_216.0, "second cancellation key")?;
    assign(&mut first_key, 2, 1.0, "third cancellation key")?;
    logical_keys.push(first_key);
    logical_keys.push(vec![0.0_f32; logical.row_elements()]);
    logical_values.push(vec![0.0_f32; logical.row_elements()]);
    logical_values.push(vec![10.0_f32; logical.row_elements()]);
    let table = vec![0_u32];
    let mut keys = vec![0.0_f32; native.key_value_elements()];
    let mut values = vec![0.0_f32; native.key_value_elements()];
    let key_elements = keys.len();
    let value_elements = values.len();
    for token in 0..logical.visible_tokens() {
        let row_start = token.checked_mul(logical.row_elements()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "cancellation fixture row start",
            },
        )?;
        let row_end = row_start.checked_add(logical.row_elements()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "cancellation physical row end",
            },
        )?;
        let key = logical_keys
            .get(token)
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "cancellation logical key row",
            })?;
        let value = logical_values
            .get(token)
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "cancellation logical value row",
            })?;
        keys.get_mut(row_start..row_end)
            .ok_or(NativeReferenceError::Extent {
                input: "keys",
                expected: native.key_value_elements(),
                actual: key_elements,
            })?
            .copy_from_slice(key);
        values
            .get_mut(row_start..row_end)
            .ok_or(NativeReferenceError::Extent {
                input: "values",
                expected: native.key_value_elements(),
                actual: value_elements,
            })?
            .copy_from_slice(value);
    }
    Ok(NativeFixture {
        native,
        query,
        logical_keys,
        logical_values,
        keys,
        values,
        table,
    })
}

fn native_f32_reference(
    native: &NativePagedDecodePlan,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    table: &[u32],
) -> Result<Vec<f32>, NativeReferenceError> {
    native_f32_reference_with_mode(
        native,
        query,
        keys,
        values,
        table,
        NativeReferenceMode::NATIVE,
    )
}

fn native_f32_reference_with_mode(
    native: &NativePagedDecodePlan,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    table: &[u32],
    mode: NativeReferenceMode,
) -> Result<Vec<f32>, NativeReferenceError> {
    validate_extent("query", query.len(), native.query_elements())?;
    validate_extent("keys", keys.len(), native.key_value_elements())?;
    validate_extent("values", values.len(), native.key_value_elements())?;
    validate_extent("page table", table.len(), native.page_table_entries())?;
    validate_native_inputs("query", query)?;
    validate_native_inputs("keys", keys)?;
    validate_native_inputs("values", values)?;
    let logical = native.logical();
    let mut output = vec![0.0_f32; native.output_elements()];
    let output_elements = output.len();
    for query_head in 0..logical.query_heads() {
        let kv_head = query_head / logical.gqa_group();
        let query_start = query_head.checked_mul(logical.head_width()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "query head start",
            },
        )?;
        let query_end = query_start.checked_add(logical.head_width()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "query head end",
            },
        )?;
        let query_head_values =
            query
                .get(query_start..query_end)
                .ok_or(NativeReferenceError::Extent {
                    input: "query",
                    expected: native.query_elements(),
                    actual: query.len(),
                })?;
        let output_head =
            output
                .get_mut(query_start..query_end)
                .ok_or(NativeReferenceError::Extent {
                    input: "output",
                    expected: native.output_elements(),
                    actual: output_elements,
                })?;
        native_f32_head(
            native,
            query_head_values,
            kv_head,
            keys,
            values,
            table,
            output_head,
            mode,
        )?;
    }
    Ok(output)
}

fn native_f32_head(
    native: &NativePagedDecodePlan,
    query: &[f32],
    kv_head: usize,
    keys: &[f32],
    values: &[f32],
    table: &[u32],
    output: &mut [f32],
    mode: NativeReferenceMode,
) -> Result<(), NativeReferenceError> {
    let logical = native.logical();
    let output_elements = output.len();
    let mut maximum = f32::NEG_INFINITY;
    let mut normalizer = 0.0_f32;
    for token in 0..logical.visible_tokens() {
        let row_start = physical_row_start(*native, table, token, kv_head)?;
        let row_end = row_start.checked_add(logical.head_width()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "native row end",
            },
        )?;
        let key = keys
            .get(row_start..row_end)
            .ok_or(NativeReferenceError::Extent {
                input: "keys",
                expected: native.key_value_elements(),
                actual: keys.len(),
            })?;
        let value = values
            .get(row_start..row_end)
            .ok_or(NativeReferenceError::Extent {
                input: "values",
                expected: native.key_value_elements(),
                actual: values.len(),
            })?;
        let dot = native_dot(query, key, mode.dot_reduction, mode.include_tail, token)?;
        let score = dot * logical.scale();
        ensure_finite(score, "score", token)?;
        let next_maximum = maximum.max(score);
        ensure_finite(next_maximum, "running maximum", token)?;
        let prior_rescale = if maximum.is_infinite() {
            0.0
        } else if mode.rescale_prior_output {
            let maximum_delta = maximum - next_maximum;
            ensure_finite(maximum_delta, "maximum delta", token)?;
            let rescale = maximum_delta.exp();
            ensure_finite(rescale, "prior rescale", token)?;
            rescale
        } else {
            1.0
        };
        ensure_finite(prior_rescale, "prior rescale", token)?;
        let score_delta = score - next_maximum;
        ensure_finite(score_delta, "score delta", token)?;
        let token_weight = score_delta.exp();
        ensure_finite(token_weight, "token weight", token)?;
        let prior_normalizer = prior_rescale * normalizer;
        ensure_finite(prior_normalizer, "prior normalizer product", token)?;
        normalizer = prior_normalizer + token_weight;
        ensure_finite(normalizer, "running normalizer", token)?;
        for lane in 0..WAVE_LANES {
            let mut column = lane;
            while column < output_elements {
                let destination =
                    output
                        .get_mut(column)
                        .ok_or(NativeReferenceError::ArithmeticOverflow {
                            operation: "native output lane",
                        })?;
                let source = value.get(column).ok_or(NativeReferenceError::Extent {
                    input: "value head",
                    expected: output_elements,
                    actual: value.len(),
                })?;
                let prior_output = prior_rescale * *destination;
                ensure_finite(prior_output, "prior output product", column)?;
                let token_output = token_weight * *source;
                ensure_finite(token_output, "token output product", column)?;
                *destination = prior_output + token_output;
                ensure_finite(*destination, "running output", column)?;
                column = column.checked_add(WAVE_LANES).ok_or(
                    NativeReferenceError::ArithmeticOverflow {
                        operation: "native output lane stride",
                    },
                )?;
            }
        }
        maximum = next_maximum;
    }
    for lane in 0..WAVE_LANES {
        let mut column = lane;
        while column < output_elements {
            let destination =
                output
                    .get_mut(column)
                    .ok_or(NativeReferenceError::ArithmeticOverflow {
                        operation: "native final output lane",
                    })?;
            *destination /= normalizer;
            ensure_finite(*destination, "output", column)?;
            column =
                column
                    .checked_add(WAVE_LANES)
                    .ok_or(NativeReferenceError::ArithmeticOverflow {
                        operation: "native final output lane stride",
                    })?;
        }
    }
    Ok(())
}

fn native_dot(
    query: &[f32],
    key: &[f32],
    reduction: DotReduction,
    include_tail: bool,
    token: usize,
) -> Result<f32, NativeReferenceError> {
    match reduction {
        DotReduction::NativeTree => native_tree_dot(query, key, include_tail, token),
        DotReduction::Sequential => sequential_dot(query, key, include_tail, token),
    }
}

fn native_tree_dot(
    query: &[f32],
    key: &[f32],
    include_tail: bool,
    token: usize,
) -> Result<f32, NativeReferenceError> {
    let mut lanes = [0.0_f32; WAVE_LANES];
    for lane in 0..WAVE_LANES {
        let mut column = lane;
        while column < query.len() && (include_tail || column < WAVE_LANES) {
            let query_value = query.get(column).ok_or(NativeReferenceError::Extent {
                input: "query head",
                expected: query.len(),
                actual: query.len(),
            })?;
            let key_value = key.get(column).ok_or(NativeReferenceError::Extent {
                input: "key head",
                expected: query.len(),
                actual: key.len(),
            })?;
            let product = *query_value * *key_value;
            ensure_finite(product, "dot product", token)?;
            let lane_value =
                lanes
                    .get_mut(lane)
                    .ok_or(NativeReferenceError::ArithmeticOverflow {
                        operation: "native lane",
                    })?;
            *lane_value += product;
            ensure_finite(*lane_value, "lane dot", lane)?;
            column =
                column
                    .checked_add(WAVE_LANES)
                    .ok_or(NativeReferenceError::ArithmeticOverflow {
                        operation: "native lane stride",
                    })?;
        }
    }
    for offset in REDUCTION_OFFSETS {
        let prior = lanes;
        for lane in 0..WAVE_LANES.saturating_sub(offset) {
            let destination =
                lanes
                    .get_mut(lane)
                    .ok_or(NativeReferenceError::ArithmeticOverflow {
                        operation: "reduction destination lane",
                    })?;
            let source =
                prior
                    .get(lane + offset)
                    .ok_or(NativeReferenceError::ArithmeticOverflow {
                        operation: "reduction source lane",
                    })?;
            *destination += *source;
            ensure_finite(*destination, "tree dot", lane)?;
        }
    }
    lanes
        .first()
        .copied()
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "lane zero",
        })
}

fn sequential_dot(
    query: &[f32],
    key: &[f32],
    include_tail: bool,
    token: usize,
) -> Result<f32, NativeReferenceError> {
    let mut dot = 0.0_f32;
    for (column, (query_value, key_value)) in query.iter().zip(key).enumerate() {
        if !include_tail && column >= WAVE_LANES {
            break;
        }
        let product = *query_value * *key_value;
        ensure_finite(product, "dot product", token)?;
        dot += product;
        ensure_finite(dot, "sequential dot", column)?;
    }
    Ok(dot)
}

fn f64_logical_materialized_oracle(
    fixture: &NativeFixture,
    visible_tokens: usize,
    wrong_gqa: bool,
) -> Result<Vec<f64>, NativeReferenceError> {
    let logical = fixture.native.logical();
    if visible_tokens > logical.visible_tokens() {
        return Err(NativeReferenceError::Extent {
            input: "visible logical tokens",
            expected: logical.visible_tokens(),
            actual: visible_tokens,
        });
    }
    let scale = checked_f64("oracle head width", logical.head_width())?
        .sqrt()
        .recip();
    let mut output = vec![0.0_f64; fixture.native.output_elements()];
    let output_elements = output.len();
    for query_head in 0..logical.query_heads() {
        let correct_kv_head = query_head / logical.gqa_group();
        let kv_head = if wrong_gqa {
            (correct_kv_head + 1) % logical.kv_heads()
        } else {
            correct_kv_head
        };
        let query_start = query_head.checked_mul(logical.head_width()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "oracle query start",
            },
        )?;
        let query_end = query_start.checked_add(logical.head_width()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "oracle query end",
            },
        )?;
        let query =
            fixture
                .query
                .get(query_start..query_end)
                .ok_or(NativeReferenceError::Extent {
                    input: "oracle query",
                    expected: fixture.native.query_elements(),
                    actual: fixture.query.len(),
                })?;
        let head_start = kv_head.checked_mul(logical.head_width()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "oracle KV start",
            },
        )?;
        let head_end = head_start.checked_add(logical.head_width()).ok_or(
            NativeReferenceError::ArithmeticOverflow {
                operation: "oracle KV end",
            },
        )?;
        let mut scores = reserve("oracle scores", visible_tokens)?;
        for token in 0..visible_tokens {
            let key_row = fixture
                .logical_keys
                .get(token)
                .ok_or(NativeReferenceError::Extent {
                    input: "oracle logical keys",
                    expected: visible_tokens,
                    actual: fixture.logical_keys.len(),
                })?;
            let key = key_row
                .get(head_start..head_end)
                .ok_or(NativeReferenceError::Extent {
                    input: "oracle key row",
                    expected: logical.row_elements(),
                    actual: key_row.len(),
                })?;
            let mut dot = 0.0_f64;
            for (query_value, key_value) in query.iter().zip(key) {
                dot += f64::from(*query_value) * f64::from(*key_value);
            }
            scores.push(dot * scale);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let normalizer = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f64>();
        for column in 0..logical.head_width() {
            let mut result = 0.0_f64;
            for token in 0..visible_tokens {
                let probability = (scores.get(token).ok_or(NativeReferenceError::Extent {
                    input: "oracle scores",
                    expected: visible_tokens,
                    actual: scores.len(),
                })? - maximum)
                    .exp()
                    / normalizer;
                let value_row =
                    fixture
                        .logical_values
                        .get(token)
                        .ok_or(NativeReferenceError::Extent {
                            input: "oracle logical values",
                            expected: visible_tokens,
                            actual: fixture.logical_values.len(),
                        })?;
                let value_index = head_start.checked_add(column).ok_or(
                    NativeReferenceError::ArithmeticOverflow {
                        operation: "oracle value index",
                    },
                )?;
                let value = value_row
                    .get(value_index)
                    .ok_or(NativeReferenceError::Extent {
                        input: "oracle value row",
                        expected: logical.row_elements(),
                        actual: value_row.len(),
                    })?;
                result += probability * f64::from(*value);
            }
            let output_index = query_start.checked_add(column).ok_or(
                NativeReferenceError::ArithmeticOverflow {
                    operation: "oracle output index",
                },
            )?;
            let destination = output
                .get_mut(output_index)
                .ok_or(NativeReferenceError::Extent {
                    input: "oracle output",
                    expected: fixture.native.output_elements(),
                    actual: output_elements,
                })?;
            *destination = result;
        }
    }
    Ok(output)
}

fn physical_row_start(
    native: NativePagedDecodePlan,
    table: &[u32],
    token: usize,
    kv_head: usize,
) -> Result<usize, NativeReferenceError> {
    let logical = native.logical();
    let logical_page = token / native.page_tokens().get();
    let physical_page = *table
        .get(logical_page)
        .ok_or(NativeReferenceError::Extent {
            input: "page table",
            expected: native.page_table_entries(),
            actual: table.len(),
        })?;
    let physical_page =
        usize::try_from(physical_page).map_err(|_| NativeReferenceError::ArithmeticOverflow {
            operation: "physical page conversion",
        })?;
    if physical_page >= native.physical_pages() {
        return Err(NativeReferenceError::PageTableEntry {
            logical_page,
            physical_page: u32::try_from(physical_page).map_err(|_| {
                NativeReferenceError::ArithmeticOverflow {
                    operation: "physical page diagnostic conversion",
                }
            })?,
            physical_pages: native.physical_pages(),
        });
    }
    let in_page_token = token % native.page_tokens().get();
    let page_row = physical_page
        .checked_mul(native.page_tokens().get())
        .and_then(|base| base.checked_add(in_page_token))
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "physical page row",
        })?;
    page_row
        .checked_mul(logical.kv_heads())
        .and_then(|base| base.checked_add(kv_head))
        .and_then(|head| head.checked_mul(logical.head_width()))
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "physical row start",
        })
}

fn fixture_physical_row_start(
    table: &[u32],
    token: usize,
    page_tokens: usize,
    row_elements: usize,
    physical_pages: usize,
) -> Result<usize, NativeReferenceError> {
    let logical_page = token / page_tokens;
    let physical_page = *table
        .get(logical_page)
        .ok_or(NativeReferenceError::Extent {
            input: "fixture page table",
            expected: logical_page.checked_add(1).ok_or(
                NativeReferenceError::ArithmeticOverflow {
                    operation: "fixture logical page count",
                },
            )?,
            actual: table.len(),
        })?;
    let physical_page =
        usize::try_from(physical_page).map_err(|_| NativeReferenceError::ArithmeticOverflow {
            operation: "fixture physical page conversion",
        })?;
    if physical_page >= physical_pages {
        return Err(NativeReferenceError::PageTableEntry {
            logical_page,
            physical_page: u32::try_from(physical_page).map_err(|_| {
                NativeReferenceError::ArithmeticOverflow {
                    operation: "fixture physical page diagnostic conversion",
                }
            })?,
            physical_pages,
        });
    }
    let in_page_token = token % page_tokens;
    physical_page
        .checked_mul(page_tokens)
        .and_then(|page| page.checked_add(in_page_token))
        .and_then(|slot| slot.checked_mul(row_elements))
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "fixture physical row start",
        })
}

fn query_reusing_head_zero(fixture: &NativeFixture) -> Result<Vec<f32>, NativeReferenceError> {
    let width = fixture.native.logical().head_width();
    let source = fixture
        .query
        .get(..width)
        .ok_or(NativeReferenceError::Extent {
            input: "query head zero",
            expected: width,
            actual: fixture.query.len(),
        })?
        .to_vec();
    let mut wrong_query = fixture.query.clone();
    let wrong_query_elements = wrong_query.len();
    let reused_start = width;
    let reused_end =
        reused_start
            .checked_add(width)
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "reused query end",
            })?;
    let destination =
        wrong_query
            .get_mut(reused_start..reused_end)
            .ok_or(NativeReferenceError::Extent {
                input: "query head one",
                expected: fixture.native.query_elements(),
                actual: wrong_query_elements,
            })?;
    destination.copy_from_slice(&source);
    Ok(wrong_query)
}

fn causal_prefill_future_leak_values(
    geometry: CausalPrefillFixtureGeometry,
) -> Result<Vec<f32>, NativeReferenceError> {
    let final_visible_tokens = geometry.final_visible_tokens()?;
    let mut values = reserve("causal prefill future-leak values", final_visible_tokens)?;
    for position in 0..final_visible_tokens {
        // The final staged row is deliberately dominant. If the first query
        // sees it, its uniform zero-score average changes by orders of
        // magnitude instead of remaining the known prefix endpoint.
        let value = if position.checked_add(1) == Some(final_visible_tokens) {
            4096.0
        } else {
            checked_f32("causal prefill prefix value", position)? + 1.0
        };
        values.push(value);
    }
    Ok(values)
}

fn causal_prefill_f64_oracle(
    geometry: CausalPrefillFixtureGeometry,
    values: &[f32],
) -> Result<Vec<f64>, NativeReferenceError> {
    let per_token = geometry
        .query_heads
        .checked_mul(geometry.head_width)
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "causal prefill oracle token output",
        })?;
    let output_elements =
        geometry
            .tokens
            .checked_mul(per_token)
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "causal prefill oracle output",
            })?;
    let mut output = reserve("causal prefill f64 oracle output", output_elements)?;
    for token in 0..geometry.tokens {
        let mean = causal_prefill_f64_mean(values, geometry.visible_tokens_for(token)?)?;
        for _ in 0..per_token {
            output.push(mean);
        }
    }
    Ok(output)
}

fn causal_prefill_f64_mean(values: &[f32], visible: usize) -> Result<f64, NativeReferenceError> {
    let prefix = values.get(..visible).ok_or(NativeReferenceError::Extent {
        input: "causal prefill f64 values",
        expected: visible,
        actual: values.len(),
    })?;
    let sum = prefix.iter().copied().map(f64::from).sum::<f64>();
    Ok(sum / checked_f64("causal prefill f64 visible tokens", visible)?)
}

fn assert_close_to_f64(
    actual: &[f32],
    expected: &[f64],
    invariant: &'static str,
) -> Result<(), NativeReferenceError> {
    validate_extent("expected output", expected.len(), actual.len())?;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected
            .to_f32()
            .ok_or(NativeReferenceError::F32Narrowing {
                input: "expected output",
                index,
            })?;
        assert!(
            relative_eq!(*actual, expected, epsilon = WELL_CONDITIONED_EPSILON),
            "{invariant} at output index {index}"
        );
    }
    Ok(())
}

fn assert_close_to_f32(
    actual: &[f32],
    expected: &[f32],
    invariant: &'static str,
) -> Result<(), NativeReferenceError> {
    validate_extent("native expected output", expected.len(), actual.len())?;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            relative_eq!(*actual, *expected, epsilon = WELL_CONDITIONED_EPSILON),
            "{invariant} at output index {index}"
        );
    }
    Ok(())
}

fn assert_f32_separated(
    left: &[f32],
    right: &[f32],
    invariant: &'static str,
) -> Result<(), NativeReferenceError> {
    validate_extent("compared f32 output", left.len(), right.len())?;
    assert!(
        left.iter()
            .zip(right)
            .any(|(left, right)| (*left - *right).abs() > WELL_CONDITIONED_EPSILON),
        "{invariant} by more than the retained numerical criterion"
    );
    Ok(())
}

fn assert_f64_separated(
    left: &[f64],
    right: &[f64],
    invariant: &'static str,
) -> Result<(), NativeReferenceError> {
    validate_extent("compared f64 output", left.len(), right.len())?;
    let criterion = f64::from(WELL_CONDITIONED_EPSILON);
    assert!(
        left.iter()
            .zip(right)
            .any(|(left, right)| (*left - *right).abs() > criterion),
        "{invariant} by more than the retained numerical criterion"
    );
    Ok(())
}

fn validate_extent(
    input: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), NativeReferenceError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NativeReferenceError::Extent {
            input,
            expected,
            actual,
        })
    }
}

fn validate_native_inputs(input: &'static str, values: &[f32]) -> Result<(), NativeReferenceError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(NativeReferenceError::NonFiniteInput { input, index });
        }
        if value != 0.0 && !value.is_normal() {
            return Err(NativeReferenceError::NonNormalInput { input, index });
        }
    }
    Ok(())
}

fn ensure_finite(
    value: f32,
    stage: &'static str,
    index: usize,
) -> Result<(), NativeReferenceError> {
    if value.is_finite() {
        if value == 0.0 || value.is_normal() {
            Ok(())
        } else {
            Err(NativeReferenceError::NonNormalArithmetic { stage, index })
        }
    } else {
        Err(NativeReferenceError::NonFiniteArithmetic { stage, index })
    }
}

fn reserve<T>(allocation: &'static str, elements: usize) -> Result<Vec<T>, NativeReferenceError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|source| NativeReferenceError::Allocation {
            allocation,
            elements,
            source,
        })?;
    Ok(values)
}

fn checked_f32(input: &'static str, value: usize) -> Result<f32, NativeReferenceError> {
    value
        .to_f32()
        .filter(|value| value.is_finite())
        .ok_or(NativeReferenceError::F32Conversion { input, value })
}

fn checked_f64(input: &'static str, value: usize) -> Result<f64, NativeReferenceError> {
    value
        .to_f64()
        .filter(|value| value.is_finite())
        .ok_or(NativeReferenceError::F64Conversion { input, value })
}

fn score_signal(token: usize) -> f32 {
    match token % 5 {
        0 => 1.0,
        1 | 2 => 3.0,
        3 => 5.0,
        _ => -2.0,
    }
}

fn fixture_query_component(query_head: usize, column: usize, head_width: usize) -> f32 {
    let head_scale = match query_head {
        0 => 1.0,
        1 => 0.5,
        2 => 1.25,
        3 => 0.75,
        _ => 0.0,
    };
    if column == 0 {
        head_scale
    } else if column == WAVE_LANES && head_width > WAVE_LANES {
        head_scale * 0.5
    } else if column.checked_add(1) == Some(head_width) {
        head_scale * -0.25
    } else {
        0.0
    }
}

fn fixture_key_component(token: usize, column: usize, head_width: usize) -> f32 {
    let score = score_signal(token);
    if column == 0 {
        score
    } else if column == WAVE_LANES && head_width > WAVE_LANES {
        score * 0.5
    } else if column.checked_add(1) == Some(head_width) {
        score * -0.25
    } else {
        0.0
    }
}

fn nonidentity_physical_page(logical_page: usize) -> Result<u32, NativeReferenceError> {
    const PAGE_PERMUTATION: [u32; PHYSICAL_PAGES] = [2, 0, 1];
    PAGE_PERMUTATION
        .get(logical_page % PHYSICAL_PAGES)
        .copied()
        .ok_or(NativeReferenceError::ArithmeticOverflow {
            operation: "nonidentity page table",
        })
}

fn swap_first_two(values: &mut [u32]) -> Result<(), NativeReferenceError> {
    let value_count = values.len();
    let (first, rest) = values
        .split_first_mut()
        .ok_or(NativeReferenceError::Extent {
            input: "page table",
            expected: 2,
            actual: value_count,
        })?;
    let second = rest.first_mut().ok_or(NativeReferenceError::Extent {
        input: "page table",
        expected: 2,
        actual: value_count,
    })?;
    core::mem::swap(first, second);
    Ok(())
}

fn assign(
    values: &mut [f32],
    index: usize,
    value: f32,
    operation: &'static str,
) -> Result<(), NativeReferenceError> {
    let value_count = values.len();
    let destination = values.get_mut(index).ok_or(NativeReferenceError::Extent {
        input: operation,
        expected: index
            .checked_add(1)
            .ok_or(NativeReferenceError::ArithmeticOverflow {
                operation: "assignment expected extent",
            })?,
        actual: value_count,
    })?;
    *destination = value;
    Ok(())
}
