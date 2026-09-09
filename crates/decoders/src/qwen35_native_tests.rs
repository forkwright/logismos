//! Reserved-device whole-block witness for the unsafe native Qwen3.5 session.

use hipcore::{Device, DeviceBuffer};

use crate::qwen35::tests::{
    CanonicalHybridOracle, assert_f32_matches_f64, canonical_hybrid_fixture_with_context,
    verify_fixture,
};
use crate::{Qwen35NativeLayerPlan, Qwen35NativeLayerSessionState, Qwen35Weights};

const FULL_ATTENTION_BLOCK: usize = 3;
const FIXTURE_CONTEXT: usize = 16;
const WITNESS_STEPS: usize = 9;

/// Build deterministic, varied full-block rows without leaving the native f32 domain.
///
/// The canonical fixture stores bounded finite normal-or-zero f32 weights; these
/// small normal inputs and its bounded mRoPE positions keep this reserved
/// witness within the unsafe kernels' declared numerical domain. This is a
/// fixture precondition, not a production final-value scan or a hardware claim.
fn witness_input(position: usize, width: usize) -> Vec<f32> {
    const VALUES: [f32; 3] = [0.25, -0.5, 0.75];
    (0..width)
        .map(|index| VALUES[(position + index) % VALUES.len()])
        .collect()
}

#[test]
#[ignore = "requires an operator-reserved HIP device; source tests do not qualify hardware"]
fn reserved_device_native_full_block_matches_independent_oracle() -> core::result::Result<(), String>
{
    let fixture = canonical_hybrid_fixture_with_context(FIXTURE_CONTEXT)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let width = oracle.hidden_width()?;

    let plan = Qwen35NativeLayerPlan::try_from_weights(
        &weights,
        FULL_ATTENTION_BLOCK,
        FIXTURE_CONTEXT,
        kernels::attention::NativePageTokens::B8,
    )
    .map_err(|error| error.to_string())?;
    let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
    // SAFETY: this ignored operator-only witness supplies the plan's verified
    // weights and bounded fixture values to its qualified native device.
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
        // device. Fixture construction establishes the declared finite
        // normal-or-zero input, weight, control, and intermediate domain.
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
