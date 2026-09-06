//! Private-state transaction witness for the bounded Qwen3.5 CPU session.

use super::*;
use crate::Qwen35Weights;
use crate::qwen35::tests::{
    CanonicalHybridOracle, OracleStateSnapshot, canonical_hybrid_fixture,
    mixed_quantized_hybrid_fixture, set_f32_value, verify_fixture,
};

const STATE_ABSOLUTE_TOLERANCE: f64 = 1.0e-3;
const STATE_RELATIVE_TOLERANCE: f64 = 1.0e-4;

type ExecutionStateSnapshot = (
    usize,
    Vec<(Vec<f32>, Vec<f32>)>,
    Vec<(usize, Vec<f32>, Vec<f32>)>,
);

#[test]
fn execution_capacity_refusal_preserves_allocation_source() -> std::result::Result<(), String> {
    let error = reserve::<u8>("capacity witness", usize::MAX)
        .err()
        .ok_or("an impossible byte capacity must refuse without attempting allocation")?;
    assert!(matches!(
        &error,
        crate::Error::ExecutionAllocation {
            target: "capacity witness",
            length: usize::MAX,
            ..
        }
    ));
    let source = std::error::Error::source(&error).ok_or("allocation source was discarded")?;
    assert!(source.is::<std::collections::TryReserveError>());
    Ok(())
}

#[test]
fn late_projection_refusal_leaves_private_hybrid_state_unchanged() -> std::result::Result<(), String>
{
    let mut fixture = canonical_hybrid_fixture()?;
    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    oracle.step(&[1])?;
    let expected_state = oracle.state_for_test();
    let good_payload = verify_fixture(&fixture)?;
    let good_weights =
        Qwen35Weights::try_from_verified(&good_payload).map_err(|error| error.to_string())?;
    let mut good_execution = good_weights
        .execution(4)
        .map_err(|error| error.to_string())?;
    good_execution
        .step(&[1])
        .map_err(|error| error.to_string())?;
    assert_private_state_matches_oracle(&good_execution, &expected_state)?;

    set_f32_value(&mut fixture, "output.weight", 0, f32::NAN)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut execution = weights.execution(4).map_err(|error| error.to_string())?;
    let before = private_state_snapshot(&execution);

    let error = execution.step(&[1]);
    assert!(
        error.is_err(),
        "a non-finite output row must refuse after the hybrid blocks"
    );
    assert_eq!(
        private_state_snapshot(&execution),
        before,
        "late output refusal must not commit recurrent convolution/GDN, KV, or position"
    );
    Ok(())
}

#[test]
fn mixed_quantized_private_state_matches_f64_oracle_and_rolls_back()
-> std::result::Result<(), String> {
    let fixture = mixed_quantized_hybrid_fixture()?;
    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    oracle.step(&[1, 2])?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut execution = weights.execution(4).map_err(|error| error.to_string())?;
    execution.step(&[1, 2]).map_err(|error| error.to_string())?;
    assert_private_state_matches_oracle(&execution, &oracle.state_for_test())?;

    oracle.step(&[3])?;
    execution.step(&[3]).map_err(|error| error.to_string())?;
    assert_private_state_matches_oracle(&execution, &oracle.state_for_test())?;

    let mut rollback = weights.execution(4).map_err(|error| error.to_string())?;
    let before = private_state_snapshot(&rollback);
    assert!(
        rollback.step(&[1, u32::MAX]).is_err(),
        "an invalid second token must refuse after staging one mixed-quantized token"
    );
    assert_eq!(
        private_state_snapshot(&rollback),
        before,
        "mixed-quantized refusal must not commit recurrent, KV, or position state"
    );
    Ok(())
}

fn private_state_snapshot(execution: &Qwen35Execution<'_, '_>) -> ExecutionStateSnapshot {
    let recurrent = execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(state) => Some(state.transaction_state_for_test()),
            LayerState::Full(_) => None,
        })
        .map(|(convolution, gdn)| (convolution.to_vec(), gdn.to_vec()))
        .collect();
    let full = execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(_) => None,
            LayerState::Full(state) => {
                Some((state.tokens, state.keys.clone(), state.values.clone()))
            }
        })
        .collect();
    (execution.position, recurrent, full)
}

fn assert_private_state_matches_oracle(
    execution: &Qwen35Execution<'_, '_>,
    expected: &OracleStateSnapshot,
) -> std::result::Result<(), String> {
    assert_eq!(
        execution.position, expected.0,
        "position differs from f64 oracle"
    );
    let actual_recurrent = execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(state) => Some(state),
            LayerState::Full(_) => None,
        })
        .collect::<Vec<_>>();
    if actual_recurrent.len() != expected.1.len() {
        return Err(format!(
            "recurrent state count differs: actual {}, expected {}",
            actual_recurrent.len(),
            expected.1.len()
        ));
    }
    for (index, (actual, expected)) in actual_recurrent.iter().zip(&expected.1).enumerate() {
        let (actual_convolution, actual_gdn) = actual.transaction_state_for_test();
        assert_f32_slice_matches_f64(actual_convolution, &expected.0, "convolution", index)?;
        assert_f32_slice_matches_f64(actual_gdn, &expected.1, "GDN", index)?;
    }
    let actual_full = execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(_) => None,
            LayerState::Full(state) => Some(state),
        })
        .collect::<Vec<_>>();
    if actual_full.len() != expected.2.len() {
        return Err(format!(
            "full-attention state count differs: actual {}, expected {}",
            actual_full.len(),
            expected.2.len()
        ));
    }
    for (index, (actual, expected)) in actual_full.iter().zip(&expected.2).enumerate() {
        assert_eq!(
            actual.tokens, expected.0,
            "full-attention token count differs at state {index}"
        );
        assert_f32_slice_matches_f64(&actual.keys, &expected.1, "KV keys", index)?;
        assert_f32_slice_matches_f64(&actual.values, &expected.2, "KV values", index)?;
    }
    Ok(())
}

fn assert_f32_slice_matches_f64(
    actual: &[f32],
    expected: &[f64],
    state: &str,
    block: usize,
) -> std::result::Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{state} state width differs at block {block}"));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(*actual);
        if !actual.is_finite() || !expected.is_finite() {
            return Err(format!(
                "{state} is non-finite at block {block} index {index}: actual {actual}, expected {expected}"
            ));
        }
        let delta = (actual - expected).abs();
        let tolerance =
            STATE_ABSOLUTE_TOLERANCE + STATE_RELATIVE_TOLERANCE * expected.abs().max(actual.abs());
        if delta > tolerance {
            return Err(format!(
                "{state} differs at block {block} index {index}: actual {actual}, expected {expected}, delta {delta}, tolerance {tolerance}"
            ));
        }
    }
    Ok(())
}
