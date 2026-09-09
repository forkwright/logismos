//! Private-state transaction witness for the bounded Qwen3.5 CPU session.

use super::*;
use crate::Qwen35Weights;
use crate::qwen35::tests::{
    CanonicalHybridOracle, OracleStateSnapshot, canonical_hybrid_fixture,
    canonical_hybrid_fixture_with_context, mixed_quantized_hybrid_fixture, set_f32_value,
    verify_fixture,
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
    let before = private_state_snapshot(&execution)?;

    let error = execution.step(&[1]);
    assert!(
        error.is_err(),
        "a non-finite output row must refuse after the hybrid blocks"
    );
    assert_eq!(
        private_state_snapshot(&execution)?,
        before,
        "late output refusal must not commit recurrent convolution/GDN, KV, or position"
    );
    Ok(())
}

#[test]
fn paged_kv_crosses_the_selected_boundary_and_rolls_back_a_committed_tail()
-> std::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context(16)?;
    let tokens = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0];
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;

    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let expected = oracle.step(&tokens)?;
    let expected_state = oracle.state_for_test();
    let mut unchunked = weights.execution(16).map_err(|error| error.to_string())?;
    let actual = unchunked.step(&tokens).map_err(|error| error.to_string())?;
    assert_f32_slice_matches_f64(&actual, &expected, "paged KV boundary logits", 0)?;
    assert_private_state_matches_oracle(&unchunked, &expected_state)?;

    let mut chunked = weights.execution(16).map_err(|error| error.to_string())?;
    let mut chunked_logits = chunked
        .step(&tokens[..7])
        .map_err(|error| error.to_string())?;
    chunked_logits.extend(
        chunked
            .step(&tokens[7..8])
            .map_err(|error| error.to_string())?,
    );
    chunked_logits.extend(
        chunked
            .step(&tokens[8..])
            .map_err(|error| error.to_string())?,
    );
    assert_eq!(
        chunked_logits, actual,
        "unequal chunks must preserve token-serial logits"
    );
    assert_eq!(
        private_state_snapshot(&chunked)?,
        private_state_snapshot(&unchunked)?
    );

    let prefix = &tokens[..7];
    let suffix = &tokens[7..15];
    let mut rollback = weights.execution(16).map_err(|error| error.to_string())?;
    rollback.step(prefix).map_err(|error| error.to_string())?;
    let before = private_state_snapshot(&rollback)?;
    let mut refused = suffix.to_vec();
    refused.push(u32::MAX);
    assert!(
        rollback.step(&refused).is_err(),
        "a late invalid token must refuse after staging rows across a committed partial tail"
    );
    assert_eq!(private_state_snapshot(&rollback)?, before);
    let retry = rollback.step(suffix).map_err(|error| error.to_string())?;
    let mut control = weights.execution(16).map_err(|error| error.to_string())?;
    control.step(prefix).map_err(|error| error.to_string())?;
    let expected_retry = control.step(suffix).map_err(|error| error.to_string())?;
    assert_eq!(
        retry, expected_retry,
        "a dropped append must preserve retry equivalence"
    );
    assert_eq!(
        private_state_snapshot(&rollback)?,
        private_state_snapshot(&control)?
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
    let before = private_state_snapshot(&rollback)?;
    assert!(
        rollback.step(&[1, u32::MAX]).is_err(),
        "an invalid second token must refuse after staging one mixed-quantized token"
    );
    assert_eq!(
        private_state_snapshot(&rollback)?,
        before,
        "mixed-quantized refusal must not commit recurrent, KV, or position state"
    );
    Ok(())
}

fn private_state_snapshot(
    execution: &Qwen35Execution<'_, '_>,
) -> std::result::Result<ExecutionStateSnapshot, String> {
    let recurrent = execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(state) => Some(state.transaction_state_for_test()),
            LayerState::Full(_) => None,
        })
        .map(|(convolution, gdn)| (convolution.to_vec(), gdn.to_vec()))
        .collect();
    let pool = execution.paged_kv_pool.as_ref().ok_or_else(|| {
        "full-attention execution is missing private paged KV backing".to_string()
    })?;
    let full = execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(_) => None,
            LayerState::Full(layer) => Some(*layer),
        })
        .map(|layer| paged_layer_snapshot(pool, layer))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((execution.position, recurrent, full))
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
    let pool = execution.paged_kv_pool.as_ref().ok_or_else(|| {
        "full-attention execution is missing private paged KV backing".to_string()
    })?;
    let actual_full = execution
        .layers
        .iter()
        .filter_map(|layer| match layer {
            LayerState::Recurrent(_) => None,
            LayerState::Full(layer) => Some(*layer),
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
        let (tokens, keys, values) = paged_layer_snapshot(pool, *actual)?;
        assert_eq!(
            tokens, expected.0,
            "full-attention token count differs at state {index}"
        );
        assert_f32_slice_matches_f64(&keys, &expected.1, "KV keys", index)?;
        assert_f32_slice_matches_f64(&values, &expected.2, "KV values", index)?;
    }
    Ok(())
}

fn paged_layer_snapshot(
    pool: &PagedKvPool,
    layer: usize,
) -> std::result::Result<(usize, Vec<f32>, Vec<f32>), String> {
    let kv = pool.layer_kv(layer).map_err(|error| error.to_string())?;
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for token in 0..kv.tokens() {
        keys.extend_from_slice(kv.key_row(token).map_err(|error| error.to_string())?);
        values.extend_from_slice(kv.value_row(token).map_err(|error| error.to_string())?);
    }
    Ok((kv.tokens(), keys, values))
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
