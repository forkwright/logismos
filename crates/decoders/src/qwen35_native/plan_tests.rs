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
    };
    assert_eq!(
        demand.total().map_err(|error| error.to_string())?,
        143,
        "native byte accounting must include weights, named scratch, I/O, controls, K/V, and table"
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
    for page_tokens in [
        kernels::attention::NativePageTokens::B8,
        kernels::attention::NativePageTokens::B16,
        kernels::attention::NativePageTokens::B32,
    ] {
        let plan = DeviceFullAttentionPlan::from_weights(&weights, 3, 4, page_tokens)
            .map_err(|error| error.to_string())?;
        assert!(
            plan.bytes.total().map_err(|error| error.to_string())? > 0,
            "verified full block must derive a nonzero owned device demand"
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
    Ok(())
}
