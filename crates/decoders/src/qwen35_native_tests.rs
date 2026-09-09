//! Reserved-device witnesses for unsafe native Qwen3.5 execution.

use hipcore::{Device, DeviceBuffer};

use crate::qwen35::tests::{
    CanonicalHybridOracle, Fixture, assert_f32_matches_f64,
    canonical_hybrid_fixture_with_context_and_rotary,
    canonical_hybrid_fixture_with_invalid_embedding_operand,
    canonical_hybrid_fixture_with_invalid_output_head_operand, canonical_hybrid_fixture_with_nextn,
    verify_fixture,
};
use crate::{
    Qwen35NativeExecutionPlan, Qwen35NativeLayerPlan, Qwen35NativeLayerSessionState,
    Qwen35NativeSessionState, Qwen35Weights,
};

const FULL_ATTENTION_BLOCK: usize = 3;
const FIXTURE_CONTEXT: usize = 16;
const WITNESS_STEPS: usize = 9;
const MODEL_WITNESS_TOKENS: [u32; 9] = [2, 0, 4, 1, 3, 2, 4, 0, 1];

/// Build deterministic, varied full-block rows without leaving the native f32 domain.
///
/// The canonical fixture explicitly uses bounded finite normal-or-zero f32
/// weights, partial mRoPE, and positions below sixteen; these small normal
/// inputs are the reviewed bounded fixture precondition for the unsafe kernel
/// domain. They are not a production final-value scan or a hardware claim.
fn witness_input(position: usize, width: usize) -> Vec<f32> {
    const VALUES: [f32; 3] = [0.25, -0.5, 0.75];
    (0..width)
        .map(|index| VALUES[(position + index) % VALUES.len()])
        .collect()
}

#[test]
#[ignore = "requires an operator-reserved visible gfx1100 device 0; source tests do not qualify hardware"]
fn reserved_device_native_full_block_matches_independent_oracle() -> core::result::Result<(), String>
{
    let fixture = canonical_hybrid_fixture_with_context_and_rotary(FIXTURE_CONTEXT, Some(64))?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let width = oracle.hidden_width();

    let plan = Qwen35NativeLayerPlan::try_from_weights(
        &weights,
        FULL_ATTENTION_BLOCK,
        FIXTURE_CONTEXT,
        kernels::attention::NativePageTokens::B8,
    )
    .map_err(|error| error.to_string())?;
    let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
    // SAFETY: the operator reserves visible gfx1100 device 0 before explicitly
    // invoking this ignored witness. The plan uses verified weights and the
    // bounded fixture supplies its numerical-domain preconditions; this test
    // does not establish device qualification.
    let mut session = unsafe { plan.into_session(&device) }.map_err(|error| error.to_string())?;

    for position in 0..WITNESS_STEPS {
        let input = witness_input(position, width);
        let expected_input = input
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let expected = oracle.full_block_step(FULL_ATTENTION_BLOCK, &expected_input)?;
        let input = DeviceBuffer::from_host(&device, &input)
            .map_err(|error| format!("upload native full-block input {position}: {error}"))?;
        // SAFETY: the fresh input is a live one-row allocation on the session
        // device. The bounded fixture supplies the declared finite
        // normal-or-zero numerical-domain preconditions.
        let output = unsafe { session.step(input) }
            .map_err(|error| format!("native full-block step {position}: {error}"))?;
        let mut actual = vec![0.0_f32; width];
        output
            .copy_to_host(&mut actual)
            .map_err(|error| format!("read native full-block output {position}: {error}"))?;
        assert_f32_matches_f64(&actual, &expected, "native full-block output")?;
        if session.state() != Qwen35NativeLayerSessionState::Ready {
            return Err(format!(
                "native full-block session was not ready after completed step {position}"
            ));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an operator-reserved visible gfx1100 device 0; source tests do not qualify hardware"]
fn reserved_device_native_full_block_status_refuses_invalid_input()
-> core::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context_and_rotary(FIXTURE_CONTEXT, Some(64))?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let plan = Qwen35NativeLayerPlan::try_from_weights(
        &weights,
        FULL_ATTENTION_BLOCK,
        FIXTURE_CONTEXT,
        kernels::attention::NativePageTokens::B8,
    )
    .map_err(|error| error.to_string())?;
    let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
    // SAFETY: this ignored witness owns the verified uploads and operates only
    // on the operator-reserved gfx1100 device. The checked status allocation is
    // initialized by session construction and the invalid input is deliberate.
    let mut session = unsafe { plan.into_session(&device) }.map_err(|error| error.to_string())?;
    let input = DeviceBuffer::from_host(&device, &[f32::NAN, 0.25, -0.5])
        .map_err(|error| format!("upload invalid native full-block input: {error}"))?;
    // SAFETY: the input is a live exact row allocation. Checked arithmetic must
    // record the deliberate invalid value before logical publication.
    if unsafe { session.step(input) }.is_ok() {
        return Err("native full-block invalid input unexpectedly returned output".to_string());
    }
    if session.state() != Qwen35NativeLayerSessionState::PoisonedKnownIdle {
        return Err("native full-block status fault must leave known-idle poison".to_string());
    }
    Ok(())
}

#[test]
#[ignore = "requires an operator-reserved visible gfx1100 device 0; source tests do not qualify hardware"]
fn reserved_device_native_main_model_matches_oracle_and_excludes_nextn()
-> core::result::Result<(), String> {
    let baseline = canonical_hybrid_fixture_with_context_and_rotary(FIXTURE_CONTEXT, Some(64))?;
    native_main_model_witness(&baseline, &baseline)?;
    for auxiliary_value in [0.25, -0.75] {
        let auxiliary =
            canonical_hybrid_fixture_with_nextn(FIXTURE_CONTEXT, Some(64), auxiliary_value)?;
        native_main_model_witness(&auxiliary, &baseline)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires an operator-reserved visible gfx1100 device 0; source tests do not qualify hardware"]
fn reserved_device_resident_model_creates_unequal_context_native_sessions()
-> core::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_context_and_rotary(FIXTURE_CONTEXT, Some(64))?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let plan = Qwen35NativeExecutionPlan::try_from_weights(
        &weights,
        FIXTURE_CONTEXT,
        kernels::attention::NativePageTokens::B8,
    )
    .map_err(|error| error.to_string())?;
    let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
    // SAFETY: this ignored witness uses the operator-reserved gfx1100 device
    // and the bounded normal-or-zero fixture. It proves API ownership only,
    // not hardware qualification or physical residency.
    let model = unsafe { plan.into_model(&device) }.map_err(|error| error.to_string())?;
    let mut first = model
        .plan_session(2)
        .map_err(|error| error.to_string())?
        .into_session()
        .map_err(|error| error.to_string())?;
    let mut second = model
        .plan_session(4)
        .map_err(|error| error.to_string())?
        .into_session()
        .map_err(|error| error.to_string())?;
    let mut first_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let mut second_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;

    for token in [2_u32, 0] {
        let expected = first_oracle.step(&[token])?;
        // SAFETY: each session owns its fresh stream and mutable state while
        // retaining the one immutable resident model through completion.
        let output = unsafe { first.step(token) }.map_err(|error| error.to_string())?;
        let mut actual = vec![0.0_f32; expected.len()];
        output
            .copy_to_host(&mut actual)
            .map_err(|error| format!("read first native model output: {error}"))?;
        assert_f32_matches_f64(&actual, &expected, "first resident-model session")?;
    }
    // SAFETY: the first exact-context plan has completed both admitted
    // positions, so this preflight refusal must not submit new work.
    if unsafe { first.step(4) }.is_ok() || first.state() != Qwen35NativeSessionState::Ready {
        return Err("native first session must retain its exact two-token context".to_string());
    }
    for token in [2_u32, 0, 4] {
        let expected = second_oracle.step(&[token])?;
        // SAFETY: the second session's state starts at position zero and is
        // distinct from the first session despite sharing immutable uploads.
        let output = unsafe { second.step(token) }.map_err(|error| error.to_string())?;
        let mut actual = vec![0.0_f32; expected.len()];
        output
            .copy_to_host(&mut actual)
            .map_err(|error| format!("read second native model output: {error}"))?;
        assert_f32_matches_f64(&actual, &expected, "second resident-model session")?;
    }
    assert_eq!(first.state(), Qwen35NativeSessionState::Ready);
    assert_eq!(second.state(), Qwen35NativeSessionState::Ready);
    Ok(())
}

#[test]
#[ignore = "requires an operator-reserved visible gfx1100 device 0; source tests do not qualify hardware"]
fn reserved_device_native_main_model_status_refuses_invalid_embedding_operand()
-> core::result::Result<(), String> {
    native_main_model_status_fault_witness(
        &canonical_hybrid_fixture_with_invalid_embedding_operand()?,
        "early embedding operand",
    )
}

#[test]
#[ignore = "requires an operator-reserved visible gfx1100 device 0; source tests do not qualify hardware"]
fn reserved_device_native_main_model_status_refuses_invalid_output_head_operand()
-> core::result::Result<(), String> {
    native_main_model_status_fault_witness(
        &canonical_hybrid_fixture_with_invalid_output_head_operand()?,
        "late output-head operand",
    )
}

fn native_main_model_status_fault_witness(
    fixture: &Fixture,
    fault_stage: &str,
) -> core::result::Result<(), String> {
    let payload = verify_fixture(fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let plan = Qwen35NativeExecutionPlan::try_from_weights(
        &weights,
        1,
        kernels::attention::NativePageTokens::B8,
    )
    .map_err(|error| error.to_string())?;
    let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
    // SAFETY: this ignored witness retains the complete model bundle on the
    // operator-reserved device. The fixture deliberately violates the checked
    // numerical domain while preserving structural artifact admission.
    let mut session = unsafe { plan.into_session(&device) }.map_err(|error| error.to_string())?;
    // SAFETY: the model owns its token input and all device resources. The
    // checked status must reject the deliberate fault before publication.
    if unsafe { session.step(0) }.is_ok() {
        return Err(format!(
            "native model {fault_stage} unexpectedly returned logits"
        ));
    }
    if session.state() != Qwen35NativeSessionState::PoisonedKnownIdle {
        return Err(format!(
            "native model {fault_stage} must leave known-idle poison"
        ));
    }
    Ok(())
}

fn native_main_model_witness(
    fixture: &Fixture,
    baseline: &Fixture,
) -> core::result::Result<(), String> {
    let payload = verify_fixture(fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    // WHY: expected values always come from the independent no-auxiliary model,
    // even when the native plan binds an artifact carrying a NextN extension.
    let mut oracle = CanonicalHybridOracle::from_fixture(baseline)?;
    let plan = Qwen35NativeExecutionPlan::try_from_weights(
        &weights,
        MODEL_WITNESS_TOKENS.len(),
        kernels::attention::NativePageTokens::B8,
    )
    .map_err(|error| error.to_string())?;
    let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
    // SAFETY: this helper is reached only by the ignored operator-reserved
    // gfx1100 witness. The canonical main weights and short continuation are
    // bounded normal-or-zero fixture operands, not a production domain scan.
    let mut session = unsafe { plan.into_session(&device) }.map_err(|error| error.to_string())?;

    // SAFETY: this vocabulary refusal submits no work; all retained fixture
    // allocations satisfy the same reserved-device lifetime preconditions.
    if unsafe { session.step(u32::MAX) }.is_ok()
        || session.state() != Qwen35NativeSessionState::Ready
    {
        return Err("native invalid token must refuse without poisoning preflight".to_string());
    }
    for token in MODEL_WITNESS_TOKENS {
        let expected = oracle.step(&[token])?;
        // SAFETY: the admitted token and owned main-model fixture satisfy the
        // explicitly bounded normal-or-zero operands/intermediates contract.
        let output = unsafe { session.step(token) }.map_err(|error| error.to_string())?;
        let mut actual = vec![0.0_f32; expected.len()];
        output
            .copy_to_host(&mut actual)
            .map_err(|error| error.to_string())?;
        assert_f32_matches_f64(&actual, &expected, "native main-model continuation")?;
        if session.state() != Qwen35NativeSessionState::Ready {
            return Err("native main model must be ready after completed token".to_string());
        }
    }
    // SAFETY: context preflight is exhausted and cannot submit another token.
    if unsafe { session.step(0) }.is_ok() || session.state() != Qwen35NativeSessionState::Ready {
        return Err("native exhausted context must refuse without a new submission".to_string());
    }
    Ok(())
}
