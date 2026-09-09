//! Source-only native full-attention plan accounting tests.

use super::plan::{DeviceByteDemand, DeviceFullAttentionPlan};
use crate::Qwen35Weights;
use crate::qwen35::tests::{canonical_hybrid_fixture, verify_fixture};

#[test]
fn byte_demand_sums_every_native_category() -> core::result::Result<(), String> {
    let demand = DeviceByteDemand {
        weights: 11,
        scratch: 13,
        input: 17,
        output: 19,
        controls: 23,
        key_values: 29,
        table: 31,
        numerical_status: 37,
    };
    assert_eq!(
        demand.total().map_err(|error| error.to_string())?,
        180,
        "native byte accounting must include weights, named scratch, I/O, controls, K/V, table, and numerical status"
    );
    Ok(())
}
#[test]
fn byte_demand_refuses_aggregate_overflow() {
    let demand = DeviceByteDemand {
        weights: usize::MAX,
        scratch: 1,
        input: 0,
        output: 0,
        controls: 0,
        key_values: 0,
        table: 0,
        numerical_status: 0,
    };
    assert!(
        demand.total().is_err(),
        "native aggregate accounting must fail before allocation on overflow"
    );
}

#[test]
fn verified_full_block_plan_accepts_each_explicit_native_page_size()
-> std::result::Result<(), String> {
    let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
    let weights = Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
    let numerical_status = kernels::numerical_status::NativeNumericalStatus::byte_demand();
    for (page_tokens, key_values, base_total) in [
        (kernels::attention::NativePageTokens::B8, 32_768, 76_640),
        (kernels::attention::NativePageTokens::B16, 65_536, 109_408),
        (kernels::attention::NativePageTokens::B32, 131_072, 174_944),
    ] {
        let plan = DeviceFullAttentionPlan::from_weights(&weights, 3, 4, page_tokens)
            .map_err(|error| error.to_string())?;
        assert_eq!(
            plan.bytes.weights, 25_804,
            "four attention and three shared-finish matrices plus four norms"
        );
        assert_eq!(
            plan.bytes.scratch, 17_528,
            "eleven attention and six shared-finish workspace buffers"
        );
        assert_eq!(plan.bytes.input, 12, "one hidden f32 input row");
        assert_eq!(plan.bytes.output, 12, "one hidden f32 output row");
        // WHY: absent dimension_count defaults to D=128 in this fixture:
        // 64 pairs each require one f32 cosine and one f32 sine.
        assert_eq!(plan.bytes.controls, 512, "128 rotary f32 controls");
        assert_eq!(
            plan.bytes.table, 4,
            "four-token context uses one table entry"
        );
        assert_eq!(
            plan.bytes.key_values, key_values,
            "page selector changes only K/V backing"
        );
        assert_eq!(
            plan.bytes.numerical_status, numerical_status,
            "one session-owned sticky numerical status must be accounted exactly once"
        );
        let total = base_total
            .checked_add(numerical_status)
            .ok_or("hand-derived native byte total overflow")?;
        assert_eq!(
            plan.bytes.total().map_err(|error| error.to_string())?,
            total,
            "hand-derived native byte total"
        );
    }
    Ok(())
}

#[test]
fn verified_plan_refuses_a_recurrent_main_block() -> std::result::Result<(), String> {
    let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
    let weights = Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
    assert!(
        DeviceFullAttentionPlan::from_weights(
            &weights,
            0,
            4,
            kernels::attention::NativePageTokens::B8
        )
        .is_err(),
        "a recurrent main block cannot be presented as native full attention"
    );
    assert!(
        DeviceFullAttentionPlan::from_weights(
            &weights,
            4,
            4,
            kernels::attention::NativePageTokens::B8
        )
        .is_err(),
        "a nonexistent block cannot form a native plan"
    );
    assert!(
        DeviceFullAttentionPlan::from_weights(
            &weights,
            3,
            0,
            kernels::attention::NativePageTokens::B8
        )
        .is_err(),
        "zero context must be refused"
    );
    assert!(
        DeviceFullAttentionPlan::from_weights(
            &weights,
            3,
            5,
            kernels::attention::NativePageTokens::B8
        )
        .is_err(),
        "context beyond verified metadata must be refused"
    );
    Ok(())
}
