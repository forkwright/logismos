//! Aggregate transaction witnesses for bounded Qwen3.5 CPU executions.

use super::qwen35_execution_oracle_tests::{
    ExecutionStateSnapshot, assert_f32_slice_matches_f64, assert_private_state_matches_oracle,
    private_state_snapshot,
};
use super::*;
use crate::Qwen35Weights;
use crate::qwen35::tests::{
    CanonicalHybridOracle, canonical_hybrid_fixture_with_context, set_f32_value, verify_fixture,
};

const BATCH_CONTEXT: usize = 16;
const BATCH_SEQUENCE_COUNT: usize = 2;
const BATCH_STEP_TOKENS: usize = 3;
const CANONICAL_HIDDEN: usize = 3;
const CANONICAL_VOCABULARY: usize = 5;
const LATE_FAILURE_TOKEN: u32 = 4;

type ExecutionStateBits = (
    usize,
    Vec<(Vec<u32>, Vec<u32>)>,
    Vec<(usize, Vec<u32>, Vec<u32>)>,
);

#[test]
fn batch_execution_deinterleaves_unequal_histories_and_continues() -> std::result::Result<(), String>
{
    let fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut first_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let mut second_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let mut first = execution(&weights, Qwen35LogitSelection::AllTokens)?;
    let mut second = execution(&weights, Qwen35LogitSelection::AllTokens)?;

    advance(
        &mut first,
        &mut first_oracle,
        &[&[0, 1, 2], &[3, 4, 0], &[1]],
    )?;
    advance(&mut second, &mut second_oracle, &[&[2, 3, 4]])?;

    let first_chunk = [2, 3];
    let second_chunk = [0, 1, 2];
    let expected_first = first_oracle.step(&first_chunk)?;
    let expected_second = second_oracle.step(&second_chunk)?;
    let mut executions = [first, second];
    let logits = Qwen35Execution::plan_batch(
        &mut executions,
        &[first_chunk.as_slice(), second_chunk.as_slice()],
    )
    .map_err(|error| error.to_string())?
    .execute()
    .map_err(|error| error.to_string())?;

    assert_eq!(logits.len(), 2, "batch results must retain sequence order");
    assert_f32_slice_matches_f64(&logits[0], &expected_first, "first batch logits", 0)?;
    assert_f32_slice_matches_f64(&logits[1], &expected_second, "second batch logits", 0)?;
    assert_private_state_matches_oracle(&executions[0], &first_oracle.state_for_test())?;
    assert_private_state_matches_oracle(&executions[1], &second_oracle.state_for_test())?;

    let first_continuation = [4];
    let second_continuation = [3];
    let expected_first = first_oracle.step(&first_continuation)?;
    let expected_second = second_oracle.step(&second_continuation)?;
    let actual_first = executions[0]
        .step(&first_continuation)
        .map_err(|error| error.to_string())?;
    let actual_second = executions[1]
        .step(&second_continuation)
        .map_err(|error| error.to_string())?;
    assert_f32_slice_matches_f64(&actual_first, &expected_first, "first continuation", 0)?;
    assert_f32_slice_matches_f64(&actual_second, &expected_second, "second continuation", 0)?;
    assert_private_state_matches_oracle(&executions[0], &first_oracle.state_for_test())?;
    assert_private_state_matches_oracle(&executions[1], &second_oracle.state_for_test())?;
    Ok(())
}

#[test]
fn batch_execution_preserves_per_sequence_logit_selection() -> std::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut all_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let mut last_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let mut all = execution(&weights, Qwen35LogitSelection::AllTokens)?;
    let mut last = execution(&weights, Qwen35LogitSelection::LastToken)?;

    advance(&mut all, &mut all_oracle, &[&[0, 1]])?;
    advance(&mut last, &mut last_oracle, &[&[3]])?;
    let all_chunk = [2, 3];
    let last_chunk = [4, 0, 1];
    let all_expected = all_oracle.step(&all_chunk)?;
    let last_expected = last_oracle.step(&last_chunk)?;
    let last_start = last_expected
        .len()
        .checked_sub(CANONICAL_VOCABULARY)
        .ok_or("last-token oracle output was shorter than one vocabulary row")?;
    let mut executions = [all, last];
    let outputs = Qwen35Execution::plan_batch(
        &mut executions,
        &[all_chunk.as_slice(), last_chunk.as_slice()],
    )
    .map_err(|error| error.to_string())?
    .execute()
    .map_err(|error| error.to_string())?;

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].len(), CANONICAL_VOCABULARY * all_chunk.len());
    assert_eq!(outputs[1].len(), CANONICAL_VOCABULARY);
    assert_f32_slice_matches_f64(&outputs[0], &all_expected, "all-token batch logits", 0)?;
    assert_f32_slice_matches_f64(
        &outputs[1],
        &last_expected[last_start..],
        "last-token batch logits",
        0,
    )?;
    assert_private_state_matches_oracle(&executions[0], &all_oracle.state_for_test())?;
    assert_private_state_matches_oracle(&executions[1], &last_oracle.state_for_test())?;
    Ok(())
}

#[test]
fn batch_execution_admits_distinct_owner_context_and_step_bounds() -> std::result::Result<(), String>
{
    let fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut wider_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let mut narrower_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let wider = execution_with_bounds(
        &weights,
        BATCH_CONTEXT,
        BATCH_STEP_TOKENS,
        Qwen35LogitSelection::AllTokens,
    )?;
    let narrower = execution_with_bounds(&weights, 4, 1, Qwen35LogitSelection::LastToken)?;
    let wider_chunk = [0, 1, 2];
    let narrower_chunk = [3];
    let wider_expected = wider_oracle.step(&wider_chunk)?;
    let narrower_expected = narrower_oracle.step(&narrower_chunk)?;
    let mut executions = [wider, narrower];
    let outputs = Qwen35Execution::plan_batch(
        &mut executions,
        &[wider_chunk.as_slice(), narrower_chunk.as_slice()],
    )
    .map_err(|error| error.to_string())?
    .execute()
    .map_err(|error| error.to_string())?;

    assert_f32_slice_matches_f64(&outputs[0], &wider_expected, "wider owner logits", 0)?;
    assert_f32_slice_matches_f64(&outputs[1], &narrower_expected, "narrower owner logits", 0)?;
    assert_private_state_matches_oracle(&executions[0], &wider_oracle.state_for_test())?;
    assert_private_state_matches_oracle(&executions[1], &narrower_oracle.state_for_test())?;
    Ok(())
}

#[test]
fn batch_plan_refuses_smaller_owner_context_despite_larger_aggregate_bound()
-> std::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let wider = execution_with_bounds(
        &weights,
        BATCH_CONTEXT,
        BATCH_STEP_TOKENS,
        Qwen35LogitSelection::AllTokens,
    )?;
    let mut narrower = execution_with_bounds(&weights, 2, 2, Qwen35LogitSelection::AllTokens)?;
    narrower.step(&[3]).map_err(|error| error.to_string())?;
    let mut executions = [wider, narrower];
    let before = execution_state_bits(&executions)?;

    assert!(
        Qwen35Execution::plan_batch(&mut executions, &[&[0], &[1, 2]]).is_err(),
        "the narrower owner's context bound must not be replaced by the aggregate maximum"
    );
    assert_eq!(
        execution_state_bits(&executions)?,
        before,
        "per-owner context refusal must leave both executions bitwise unchanged"
    );
    Ok(())
}

#[test]
fn batch_plan_preflight_refusals_leave_every_execution_unchanged() -> std::result::Result<(), String>
{
    let fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;

    let mut count = [
        execution(&weights, Qwen35LogitSelection::AllTokens)?,
        execution(&weights, Qwen35LogitSelection::AllTokens)?,
    ];
    assert_batch_refusal_preserves(&mut count, &[&[0, 1]])?;

    let mut empty = [execution(&weights, Qwen35LogitSelection::AllTokens)?];
    assert_batch_refusal_preserves(&mut empty, &[&[]])?;

    let mut token = [execution(&weights, Qwen35LogitSelection::AllTokens)?];
    assert_batch_refusal_preserves(&mut token, &[&[u32::MAX]])?;

    let mut step_bound = [weights
        .execution_plan(BATCH_CONTEXT, 2, Qwen35LogitSelection::AllTokens)
        .map_err(|error| error.to_string())?
        .execution()
        .map_err(|error| error.to_string())?];
    assert_batch_refusal_preserves(&mut step_bound, &[&[0, 1, 2]])?;

    let mut context_bound = [execution(&weights, Qwen35LogitSelection::AllTokens)?];
    for chunk in [
        &[0, 1, 2][..],
        &[3, 4, 0][..],
        &[1, 2, 3][..],
        &[4, 0, 1][..],
        &[2, 3, 4][..],
    ] {
        context_bound[0]
            .step(chunk)
            .map_err(|error| error.to_string())?;
    }
    assert_batch_refusal_preserves(&mut context_bound, &[&[0, 1]])?;

    let mut changed_fixture = fixture.clone();
    set_f32_value(&mut changed_fixture, "output.weight", 0, 0.25)?;
    let changed_payload = verify_fixture(&changed_fixture)?;
    let changed_weights =
        Qwen35Weights::try_from_verified(&changed_payload).map_err(|error| error.to_string())?;
    let mut different_artifacts = [
        execution(&weights, Qwen35LogitSelection::AllTokens)?,
        execution(&changed_weights, Qwen35LogitSelection::AllTokens)?,
    ];
    assert_batch_refusal_preserves(&mut different_artifacts, &[&[0], &[1]])?;
    Ok(())
}

#[test]
fn batch_plan_aggregates_identical_artifacts_conservatively() -> std::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let shared_payload = verify_fixture(&fixture)?;
    let shared_weights =
        Qwen35Weights::try_from_verified(&shared_payload).map_err(|error| error.to_string())?;
    let cloned_shared_weights = shared_weights.clone();
    let separately_verified_payload = verify_fixture(&fixture)?;
    let separately_verified_weights =
        Qwen35Weights::try_from_verified(&separately_verified_payload)
            .map_err(|error| error.to_string())?;
    let first_plan = cloned_shared_weights
        .execution_plan(
            BATCH_CONTEXT,
            BATCH_STEP_TOKENS,
            Qwen35LogitSelection::AllTokens,
        )
        .map_err(|error| error.to_string())?;
    let second_plan = separately_verified_weights
        .execution_plan(
            BATCH_CONTEXT,
            BATCH_STEP_TOKENS,
            Qwen35LogitSelection::LastToken,
        )
        .map_err(|error| error.to_string())?;
    let first_requirements = first_plan.cpu_requirements();
    let second_requirements = second_plan.cpu_requirements();
    assert_eq!(
        first_requirements.artifact_digest(),
        second_requirements.artifact_digest(),
        "separately verified identical bytes must retain their content identity"
    );
    let mut executions = [
        first_plan.execution().map_err(|error| error.to_string())?,
        second_plan.execution().map_err(|error| error.to_string())?,
    ];
    let plan = Qwen35Execution::plan_batch(&mut executions, &[&[0, 1], &[2]])
        .map_err(|error| error.to_string())?;
    let requirements = plan.cpu_requirements();

    assert_eq!(
        requirements.artifact_digest(),
        first_requirements.artifact_digest()
    );
    assert_eq!(requirements.sequence_count(), BATCH_SEQUENCE_COUNT);
    assert_eq!(requirements.total_tokens(), 3);
    assert_eq!(requirements.max_context(), BATCH_CONTEXT);
    assert_eq!(
        requirements.serialized_backing_upper_bound_bytes(),
        checked_sum(
            &[
                first_requirements.serialized_backing_bytes(),
                second_requirements.serialized_backing_bytes(),
            ],
            "serialized backing upper bound",
        )?
    );
    assert_eq!(
        requirements.retained_bytes(),
        checked_sum(
            &[
                first_requirements.retained_bytes(),
                second_requirements.retained_bytes(),
            ],
            "retained backing",
        )?
    );
    assert_eq!(
        requirements.transaction_copy_bytes(),
        checked_sum(
            &[
                first_requirements.transaction_copy_bytes(),
                second_requirements.transaction_copy_bytes(),
            ],
            "transaction-copy backing",
        )?
    );
    assert_eq!(
        requirements.workspace_upper_bound_bytes(),
        first_requirements
            .workspace_upper_bound_bytes()
            .max(second_requirements.workspace_upper_bound_bytes())
    );
    assert_eq!(
        requirements.returned_logits_bytes(),
        checked_sum(
            &[
                first_requirements.returned_logits_bytes(),
                second_requirements.returned_logits_bytes(),
            ],
            "returned logits",
        )?
    );
    assert_eq!(
        requirements.logical_f32_upper_bound_bytes(),
        checked_sum(
            &[
                requirements.retained_bytes(),
                requirements.transaction_copy_bytes(),
                requirements.workspace_upper_bound_bytes(),
                requirements.returned_logits_bytes(),
            ],
            "logical batch upper bound",
        )?
    );
    drop(plan);
    Ok(())
}

#[test]
fn dropped_batch_plan_does_not_publish_private_state() -> std::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut executions = [
        execution(&weights, Qwen35LogitSelection::AllTokens)?,
        execution(&weights, Qwen35LogitSelection::AllTokens)?,
    ];
    executions[0]
        .step(&[0, 1])
        .map_err(|error| error.to_string())?;
    executions[1]
        .step(&[2])
        .map_err(|error| error.to_string())?;
    let before = execution_state_bits(&executions)?;
    let plan = Qwen35Execution::plan_batch(&mut executions, &[&[2, 3], &[4]])
        .map_err(|error| error.to_string())?;
    drop(plan);
    assert_eq!(
        execution_state_bits(&executions)?,
        before,
        "planning without execution must not publish staged state"
    );
    Ok(())
}

#[test]
fn late_second_sequence_refusal_rolls_back_the_whole_batch_and_retries()
-> std::result::Result<(), String> {
    let mut faulty_fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let faulty_row = usize::try_from(LATE_FAILURE_TOKEN).map_err(|error| error.to_string())?;
    let faulty_index = faulty_row
        .checked_mul(CANONICAL_HIDDEN)
        .ok_or("late-failure embedding offset overflowed")?;
    set_f32_value(
        &mut faulty_fixture,
        "token_embd.weight",
        faulty_index,
        f32::NAN,
    )?;
    let faulty_payload = verify_fixture(&faulty_fixture)?;
    let faulty_weights =
        Qwen35Weights::try_from_verified(&faulty_payload).map_err(|error| error.to_string())?;
    let mut executions = [
        execution(&faulty_weights, Qwen35LogitSelection::AllTokens)?,
        execution(&faulty_weights, Qwen35LogitSelection::AllTokens)?,
    ];
    executions[0]
        .step(&[0, 1, 2])
        .map_err(|error| error.to_string())?;
    executions[1]
        .step(&[3])
        .map_err(|error| error.to_string())?;
    let before = execution_state_bits(&executions)?;
    let first_chunk = [2, 3];
    let second_chunk = [0, LATE_FAILURE_TOKEN];
    let error = Qwen35Execution::plan_batch(
        &mut executions,
        &[first_chunk.as_slice(), second_chunk.as_slice()],
    )
    .map_err(|error| error.to_string())?
    .execute()
    .err()
    .ok_or("the second sequence's non-finite embedding unexpectedly executed")?;
    assert!(
        matches!(
            &error,
            crate::Error::ProjectionRow {
                name,
                row,
                source: quant::Error::NonFiniteF32Weight { index: 0, .. },
                ..
            } if name == TOKEN_EMBEDDING && *row == faulty_row
        ),
        "the valid token id must fail only when its unique embedding row executes: {error}"
    );
    assert_eq!(
        execution_state_bits(&executions)?,
        before,
        "late second-sequence failure must not publish either staged state"
    );

    let retry_second = [0, 1];
    let retry = Qwen35Execution::plan_batch(
        &mut executions,
        &[first_chunk.as_slice(), retry_second.as_slice()],
    )
    .map_err(|error| error.to_string())?
    .execute()
    .map_err(|error| error.to_string())?;
    let good_fixture = canonical_hybrid_fixture_with_context(BATCH_CONTEXT)?;
    let good_payload = verify_fixture(&good_fixture)?;
    let good_weights =
        Qwen35Weights::try_from_verified(&good_payload).map_err(|error| error.to_string())?;
    let mut control = [
        execution(&good_weights, Qwen35LogitSelection::AllTokens)?,
        execution(&good_weights, Qwen35LogitSelection::AllTokens)?,
    ];
    control[0]
        .step(&[0, 1, 2])
        .map_err(|error| error.to_string())?;
    control[1].step(&[3]).map_err(|error| error.to_string())?;
    let expected = Qwen35Execution::plan_batch(
        &mut control,
        &[first_chunk.as_slice(), retry_second.as_slice()],
    )
    .map_err(|error| error.to_string())?
    .execute()
    .map_err(|error| error.to_string())?;
    assert_eq!(
        logits_bits(retry),
        logits_bits(expected),
        "whole-batch retry must match a pristine control bitwise"
    );
    assert_eq!(
        execution_state_bits(&executions)?,
        execution_state_bits(&control)?,
        "retry must publish exactly the pristine-control state"
    );
    Ok(())
}

fn execution(
    weights: &Qwen35Weights,
    selection: Qwen35LogitSelection,
) -> std::result::Result<Qwen35Execution, String> {
    execution_with_bounds(weights, BATCH_CONTEXT, BATCH_STEP_TOKENS, selection)
}

fn execution_with_bounds(
    weights: &Qwen35Weights,
    max_context: usize,
    max_step_tokens: usize,
    selection: Qwen35LogitSelection,
) -> std::result::Result<Qwen35Execution, String> {
    weights
        .execution_plan(max_context, max_step_tokens, selection)
        .map_err(|error| error.to_string())?
        .execution()
        .map_err(|error| error.to_string())
}

fn advance(
    execution: &mut Qwen35Execution,
    oracle: &mut CanonicalHybridOracle,
    chunks: &[&[u32]],
) -> std::result::Result<(), String> {
    for chunk in chunks {
        execution.step(chunk).map_err(|error| error.to_string())?;
        oracle.step(chunk)?;
    }
    Ok(())
}

fn assert_batch_refusal_preserves(
    executions: &mut [Qwen35Execution],
    token_ids: &[&[u32]],
) -> std::result::Result<(), String> {
    let before = execution_state_bits(executions)?;
    assert!(
        Qwen35Execution::plan_batch(executions, token_ids).is_err(),
        "invalid batch admission unexpectedly succeeded"
    );
    assert_eq!(
        execution_state_bits(executions)?,
        before,
        "batch preflight refusal must leave every execution untouched"
    );
    Ok(())
}

fn execution_state_bits(
    executions: &[Qwen35Execution],
) -> std::result::Result<Vec<ExecutionStateBits>, String> {
    executions
        .iter()
        .map(|execution| private_state_snapshot(execution).map(snapshot_bits))
        .collect()
}

fn snapshot_bits(snapshot: ExecutionStateSnapshot) -> ExecutionStateBits {
    let (position, recurrent, full) = snapshot;
    (
        position,
        recurrent
            .into_iter()
            .map(|(convolution, gdn)| (f32_bits(convolution), f32_bits(gdn)))
            .collect(),
        full.into_iter()
            .map(|(tokens, keys, values)| (tokens, f32_bits(keys), f32_bits(values)))
            .collect(),
    )
}

fn f32_bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

fn logits_bits(logits: Vec<Vec<f32>>) -> Vec<Vec<u32>> {
    logits.into_iter().map(f32_bits).collect()
}

fn checked_sum(values: &[u64], context: &'static str) -> std::result::Result<u64, String> {
    values.iter().copied().try_fold(0_u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| format!("{context} overflowed"))
    })
}
