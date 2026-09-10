//! Reserved-device witness for atomic native Qwen3.5 mixed-sequence batches.

use hipcore::Device;

use crate::qwen35::tests::{
    CanonicalHybridOracle, Fixture, canonical_hybrid_fixture_with_context_and_rotary,
    verify_fixture,
};
use crate::qwen35_native_tests::{
    FIXTURE_CONTEXT, MODEL_PREFILL_CAPACITY, assert_native_prefill_terminal,
    assert_native_terminal_logits,
};
use crate::{
    Qwen35NativeExecutionPlan, Qwen35NativeExecutionSession, Qwen35Weights,
};

const SECOND_SESSION_CONTEXT: usize = 8;
const FIRST_HISTORY: [u32; 3] = [2, 0, 4];
const FIRST_PARTIAL_TAIL: [u32; 2] = [1, 3];
const SECOND_HISTORY: [u32; 2] = [4, 1];
const PAGE_EDGE_CHUNK: [u32; 3] = [2, 4, 0];
const SHORT_CHUNK: [u32; 2] = [3, 2];
const FIRST_CONTINUATION: u32 = 1;
const SECOND_CONTINUATION: u32 = 0;

type NativeMixedSessions = [Qwen35NativeExecutionSession; 2];

#[test]
#[ignore = "requires an operator-reserved visible gfx1100 device 0; source tests do not qualify hardware"]
fn reserved_device_native_mixed_sequence_batch_matches_oracle_and_continues()
-> core::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context_and_rotary(FIXTURE_CONTEXT, Some(64))?;
    let mut sessions = build_native_mixed_sessions(&fixture)?;
    let mut first_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let mut second_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;

    prime_native_mixed_sessions(&mut sessions, &mut first_oracle, &mut second_oracle)?;
    assert_native_mixed_batch_terminals(&mut sessions, &mut first_oracle, &mut second_oracle)?;
    assert_native_mixed_batch_continuations(&mut sessions, &mut first_oracle, &mut second_oracle)
}

fn build_native_mixed_sessions(
    fixture: &Fixture,
) -> core::result::Result<NativeMixedSessions, String> {
    let payload = verify_fixture(fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let plan = Qwen35NativeExecutionPlan::try_from_weights_prefill(
        &weights,
        FIXTURE_CONTEXT,
        MODEL_PREFILL_CAPACITY,
        kernels::attention::NativePageTokens::B8,
    )
    .map_err(|error| error.to_string())?;
    let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
    // SAFETY: this ignored witness owns the bounded synthetic fixture and runs
    // only after an operator reserves the visible gfx1100 device.
    let model = unsafe { plan.into_model(&device) }.map_err(|error| error.to_string())?;
    let first = model
        .plan_prefill_session(FIXTURE_CONTEXT, MODEL_PREFILL_CAPACITY)
        .map_err(|error| error.to_string())?
        .into_session()
        .map_err(|error| error.to_string())?;
    let second = model
        .plan_prefill_session(SECOND_SESSION_CONTEXT, MODEL_PREFILL_CAPACITY)
        .map_err(|error| error.to_string())?
        .into_session()
        .map_err(|error| error.to_string())?;
    Ok([first, second])
}

fn prime_native_mixed_sessions(
    sessions: &mut NativeMixedSessions,
    first_oracle: &mut CanonicalHybridOracle,
    second_oracle: &mut CanonicalHybridOracle,
) -> core::result::Result<(), String> {
    assert_native_prefill_terminal(
        &mut sessions[0],
        first_oracle,
        &FIRST_HISTORY,
        "first mixed-batch history",
    )?;
    assert_native_prefill_terminal(
        &mut sessions[0],
        first_oracle,
        &FIRST_PARTIAL_TAIL,
        "first partial tail",
    )?;
    assert_native_prefill_terminal(
        &mut sessions[1],
        second_oracle,
        &SECOND_HISTORY,
        "second mixed-batch history",
    )?;
    Ok(())
}

fn assert_native_mixed_batch_terminals(
    sessions: &mut NativeMixedSessions,
    first_oracle: &mut CanonicalHybridOracle,
    second_oracle: &mut CanonicalHybridOracle,
) -> core::result::Result<(), String> {
    let first_expected = first_oracle.step(&PAGE_EDGE_CHUNK)?;
    let second_expected = second_oracle.step(&SHORT_CHUNK)?;
    // SAFETY: both mutable sessions retain the same resident model Arc, while
    // their independent state, chunk capacities, and contexts are already
    // admitted. The first sequence reaches the B8 page edge from a partial
    // tail; the second contributes a shorter independent chunk.
    let outputs = unsafe {
        Qwen35NativeExecutionSession::plan_batch(
            sessions,
            &[PAGE_EDGE_CHUNK.as_slice(), SHORT_CHUNK.as_slice()],
        )
        .map_err(|error| error.to_string())?
        .execute()
    }
    .map_err(|error| error.to_string())?;
    if outputs.len() != sessions.len() {
        return Err("native mixed batch must return one result per input sequence".to_string());
    }
    assert_native_terminal_logits(
        &outputs[0],
        &first_expected,
        PAGE_EDGE_CHUNK.len(),
        "first mixed-batch terminal logits",
    )?;
    assert_native_terminal_logits(
        &outputs[1],
        &second_expected,
        SHORT_CHUNK.len(),
        "second mixed-batch terminal logits",
    )?;
    drop(outputs);
    Ok(())
}

fn assert_native_mixed_batch_continuations(
    sessions: &mut NativeMixedSessions,
    first_oracle: &mut CanonicalHybridOracle,
    second_oracle: &mut CanonicalHybridOracle,
) -> core::result::Result<(), String> {
    let first_expected = first_oracle.step(&[FIRST_CONTINUATION])?;
    let second_expected = second_oracle.step(&[SECOND_CONTINUATION])?;
    // SAFETY: the completed aggregate transaction has published each sequence
    // exactly once, so these T1 calls observe its committed private state.
    let first_output = unsafe { sessions[0].step(FIRST_CONTINUATION) }
        .map_err(|error| error.to_string())?;
    // SAFETY: this is the second independently owned session's T1 continuation.
    let second_output = unsafe { sessions[1].step(SECOND_CONTINUATION) }
        .map_err(|error| error.to_string())?;
    assert_native_terminal_logits(
        &first_output,
        &first_expected,
        1,
        "first mixed-batch continuation",
    )?;
    assert_native_terminal_logits(
        &second_output,
        &second_expected,
        1,
        "second mixed-batch continuation",
    )?;
    Ok(())
}
