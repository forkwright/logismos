//! Private-state transaction witness for the bounded Qwen3.5 CPU session.

use super::*;
use crate::Qwen35Weights;
use crate::qwen35::tests::{
    CanonicalHybridOracle, canonical_hybrid_fixture, set_f32_value, verify_fixture,
};

type ExecutionStateSnapshot = (
    usize,
    Vec<(Vec<f32>, Vec<f32>)>,
    Vec<(usize, Vec<f32>, Vec<f32>)>,
);
type OracleStateSnapshot = (usize, Vec<(Vec<f64>, Vec<f64>)>, Vec<f64>, Vec<f64>);

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
    for (index, (actual, expected)) in execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(state) => Some(state),
            LayerState::Full(_) => None,
        })
        .zip(&expected.1)
        .enumerate()
    {
        let (actual_convolution, actual_gdn) = actual.transaction_state_for_test();
        assert_f32_slice_matches_f64(actual_convolution, &expected.0, "convolution", index)?;
        assert_f32_slice_matches_f64(actual_gdn, &expected.1, "GDN", index)?;
    }
    let actual_full = execution
        .layers
        .iter()
        .find_map(|layer| match layer {
            LayerState::Recurrent(_) => None,
            LayerState::Full(state) => Some(state),
        })
        .ok_or_else(|| "full KV state is absent".to_string())?;
    assert_f32_slice_matches_f64(&actual_full.keys, &expected.2, "KV keys", 3)?;
    assert_f32_slice_matches_f64(&actual_full.values, &expected.3, "KV values", 3)
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
        if (f64::from(*actual) - expected).abs() > 2.0e-5 {
            return Err(format!("{state} differs at block {block} index {index}"));
        }
    }
    Ok(())
}
