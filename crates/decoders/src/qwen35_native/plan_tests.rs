//! Source-only native full-attention plan accounting tests.

use super::plan::DeviceByteDemand;

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
