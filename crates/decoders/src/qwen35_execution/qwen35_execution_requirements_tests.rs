//! Plan-derived CPU logical-capacity witnesses.

use super::*;
use crate::Qwen35Weights;
use crate::qwen35::Qwen35RecurrentLayout;
use crate::qwen35::tests::{canonical_hybrid_fixture, verify_fixture};
use crate::qwen35_requirements::Qwen35RequirementElements;

#[test]
fn named_owners_reconcile_a_large_nondegenerate_shape() -> std::result::Result<(), String> {
    let layout = demanding_layout();
    let recurrent_layout = demanding_recurrent_layout();

    let convolution = kernels::CausalConvAllocationPlan::try_from_dimensions(1, 110, 4)
        .map_err(|error| error.to_string())?;
    assert_eq!(convolution.output_elements(), 110);
    assert_eq!(convolution.history_elements(), 330);

    let gdn = kernels::MultiHeadRecurrentAllocationPlan::try_from_dimensions(1, 6, 6, 17, 7)
        .map_err(|error| error.to_string())?;
    assert_eq!(gdn.output_elements(), 42);
    assert_eq!(gdn.state_elements(), 714);
    assert_eq!(gdn.head_output_elements(), 7);
    assert_eq!(gdn.head_state_elements(), 119);
    assert_eq!(gdn.state_times_key_elements(), 7);
    assert_eq!(gdn.delta_elements(), 7);
    assert_eq!(gdn.workspace_elements(), 896);

    assert_eq!(
        kernels::cpu_f32::rms_norm_output_elements(6, 7).map_err(|error| error.to_string())?,
        42
    );
    assert_eq!(kernels::cpu_f32::unary_output_elements(42), 42);
    assert_eq!(kernels::cpu_f32::binary_output_elements(42), 42);

    let elements = Qwen35RequirementElements::try_from_layout(
        layout,
        recurrent_layout,
        17,
        Qwen35LogitSelection::AllTokens,
        paged_kv_plan(layout).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    // Independently hand-computed from the named requests. This shape keeps
    // recurrent KxV state, convolution history, context, FFN, and vocabulary
    // distinct so no hidden-width coefficient can accidentally satisfy it.
    assert_eq!(elements.recurrent_layer_retained, 1_510);
    assert_eq!(elements.full_attention_pool_retained, 5_440);
    assert_eq!(elements.retained, 11_480);
    assert_eq!(elements.transaction_copy, 6_040);
    assert_eq!(elements.recurrent_workspace, 2_704);
    assert_eq!(elements.full_attention_workspace, 496);
    assert_eq!(elements.layer_finish_workspace, 104);
    assert_eq!(elements.lm_head_workspace, 1_023);
    assert_eq!(elements.workspace_upper_bound, 2_711);
    assert_eq!(elements.returned_logits, 17_153);
    assert_eq!(elements.logical_upper_bound, 37_384);

    let last = Qwen35RequirementElements::try_from_layout(
        layout,
        recurrent_layout,
        17,
        Qwen35LogitSelection::LastToken,
        paged_kv_plan(layout).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    assert_eq!(last.returned_logits, 1_009);
    assert_eq!(last.logical_upper_bound, 21_240);
    assert_eq!(last.retained, elements.retained);
    assert_eq!(last.transaction_copy, elements.transaction_copy);
    assert_eq!(last.workspace_upper_bound, elements.workspace_upper_bound);
    Ok(())
}

#[test]
fn independent_shape_axes_change_their_named_owner_components() -> std::result::Result<(), String> {
    let layout = demanding_layout();
    let recurrent = demanding_recurrent_layout();
    let baseline = requirement_elements(layout, recurrent)?;

    let kernel = requirement_elements(
        layout,
        Qwen35RecurrentLayout {
            conv_kernel: 5,
            ..recurrent
        },
    )?;
    assert!(kernel.recurrent_layer_retained > baseline.recurrent_layer_retained);
    assert!(kernel.recurrent_workspace > baseline.recurrent_workspace);

    let state = requirement_elements(
        layout,
        Qwen35RecurrentLayout {
            state: 19,
            ..recurrent
        },
    )?;
    assert!(state.recurrent_layer_retained > baseline.recurrent_layer_retained);
    assert!(state.recurrent_workspace > baseline.recurrent_workspace);

    let values = requirement_elements(
        layout,
        Qwen35RecurrentLayout {
            inner: 48,
            ..recurrent
        },
    )?;
    assert!(values.recurrent_layer_retained > baseline.recurrent_layer_retained);
    assert!(values.recurrent_workspace > baseline.recurrent_workspace);

    let ffn = requirement_elements(
        Layout {
            feed_forward: 97,
            ..layout
        },
        recurrent,
    )?;
    assert!(ffn.layer_finish_workspace > baseline.layer_finish_workspace);

    let vocabulary = requirement_elements(
        Layout {
            vocabulary: 2_003,
            ..layout
        },
        recurrent,
    )?;
    assert!(vocabulary.lm_head_workspace > baseline.lm_head_workspace);
    assert!(vocabulary.returned_logits > baseline.returned_logits);

    let context = requirement_elements(
        Layout {
            max_context: 521,
            ..layout
        },
        recurrent,
    )?;
    assert!(context.full_attention_pool_retained > baseline.full_attention_pool_retained);
    assert!(context.full_attention_workspace > baseline.full_attention_workspace);
    Ok(())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one integration witness reconciles report identity, categories, output shape, and committed state"
)]
fn artifact_plan_report_and_all_last_execution_agree() -> std::result::Result<(), String> {
    let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
    let weights = Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
    let all = weights
        .execution_plan(4, 3, Qwen35LogitSelection::AllTokens)
        .map_err(|error| error.to_string())?;
    let last = weights
        .execution_plan(4, 3, Qwen35LogitSelection::LastToken)
        .map_err(|error| error.to_string())?;
    let vocabulary = all.layout.vocabulary;
    let all_requirements = all.cpu_requirements();
    let last_requirements = last.cpu_requirements();

    assert_eq!(
        all_requirements.artifact_digest(),
        artifact.observation().inspection().digest
    );
    assert_eq!(
        all_requirements.serialized_backing_bytes(),
        artifact.observation().inspection().file_len
    );
    assert_eq!(all_requirements.max_context(), 4);
    assert_eq!(all_requirements.max_step_tokens(), 3);
    assert_eq!(
        all_requirements.selection(),
        Qwen35LogitSelection::AllTokens
    );
    assert_eq!(
        last_requirements.selection(),
        Qwen35LogitSelection::LastToken
    );
    assert_eq!(
        all_requirements.retained_bytes(),
        last_requirements.retained_bytes()
    );
    assert_eq!(
        all_requirements.transaction_copy_bytes(),
        last_requirements.transaction_copy_bytes()
    );
    assert_eq!(
        all_requirements.workspace_upper_bound_bytes(),
        last_requirements.workspace_upper_bound_bytes()
    );
    assert!(
        all_requirements.transaction_copy_bytes() < all_requirements.retained_bytes(),
        "paged KV backing is retained once and must not be cloned into the recurrent transaction"
    );
    assert_eq!(
        all_requirements.returned_logits_bytes(),
        last_requirements
            .returned_logits_bytes()
            .checked_mul(3)
            .ok_or_else(|| "test returned-logit byte multiplication overflowed".to_string())?
    );
    assert_eq!(
        all_requirements.logical_f32_upper_bound_bytes(),
        report_component_sum(all_requirements)?
    );
    assert_eq!(
        last_requirements.logical_f32_upper_bound_bytes(),
        report_component_sum(last_requirements)?
    );

    let mut all_execution = all.execution().map_err(|error| error.to_string())?;
    let mut last_execution = last.execution().map_err(|error| error.to_string())?;
    let all_logits = all_execution
        .step(&[0, 1, 2])
        .map_err(|error| error.to_string())?;
    let last_logits = last_execution
        .step(&[0, 1, 2])
        .map_err(|error| error.to_string())?;
    assert_eq!(all_logits.len(), vocabulary * 3);
    assert_eq!(last_logits.len(), vocabulary);
    assert_eq!(all_execution.position, 3);
    assert_eq!(last_execution.position, 3);
    assert_same_execution_state(&all_execution, &last_execution)?;

    all_execution
        .step(&[3])
        .map_err(|error| error.to_string())?;
    last_execution
        .step(&[3])
        .map_err(|error| error.to_string())?;
    assert_eq!(all_execution.position, 4);
    assert_eq!(last_execution.position, 4);
    assert_same_execution_state(&all_execution, &last_execution)?;
    assert!(all_execution.step(&[0]).is_err());
    assert!(last_execution.step(&[0]).is_err());
    assert_eq!(all_execution.position, 4);
    assert_eq!(last_execution.position, 4);
    Ok(())
}

#[test]
fn owner_arithmetic_overflow_is_rejected_before_execution() -> std::result::Result<(), String> {
    assert!(kernels::CausalConvAllocationPlan::try_from_dimensions(usize::MAX, 2, 4).is_err());
    assert!(
        kernels::MultiHeadRecurrentAllocationPlan::try_from_dimensions(usize::MAX, 2, 2, 2, 2,)
            .is_err()
    );

    let overflowing = Layout {
        max_context: usize::MAX,
        ..demanding_layout()
    };
    let error = Qwen35RequirementElements::try_from_layout(
        overflowing,
        demanding_recurrent_layout(),
        1,
        Qwen35LogitSelection::AllTokens,
        paged_kv_plan(demanding_layout()).map_err(|error| error.to_string())?,
    )
    .err()
    .ok_or_else(|| "overflowing owner plan unexpectedly succeeded".to_string())?;
    assert!(matches!(error, crate::Error::ArithmeticOverflow { .. }));
    Ok(())
}

fn demanding_layout() -> Layout {
    Layout {
        hidden: 7,
        hidden_u64: 7,
        feed_forward: 19,
        heads: 6,
        kv_heads: 2,
        key: 5,
        n_rot: 4,
        key_u64: 5,
        kv_width: 10,
        query_width: 30,
        gqa_group: 3,
        vocabulary: 1_009,
        main_blocks: 5,
        full_interval: 3,
        max_context: 257,
        epsilon: 1.0e-5,
        rope_base: 10_000.0,
        rope_sections: [1, 1, 0, 0],
    }
}

fn demanding_recurrent_layout() -> Qwen35RecurrentLayout {
    Qwen35RecurrentLayout {
        hidden: 7,
        conv_kernel: 4,
        inner: 42,
        state: 17,
        time_step_rank: 6,
        group_count: 2,
        main_block_count: 5,
        full_attention_interval: 3,
    }
}

fn requirement_elements(
    layout: Layout,
    recurrent: Qwen35RecurrentLayout,
) -> std::result::Result<Qwen35RequirementElements, String> {
    Qwen35RequirementElements::try_from_layout(
        layout,
        recurrent,
        17,
        Qwen35LogitSelection::AllTokens,
        paged_kv_plan(layout).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn report_component_sum(
    requirements: crate::Qwen35CpuRequirements,
) -> std::result::Result<u64, String> {
    requirements
        .retained_bytes()
        .checked_add(requirements.transaction_copy_bytes())
        .and_then(|sum| sum.checked_add(requirements.workspace_upper_bound_bytes()))
        .and_then(|sum| sum.checked_add(requirements.returned_logits_bytes()))
        .ok_or_else(|| "test report component sum overflowed".to_string())
}

fn assert_same_execution_state(
    left_execution: &Qwen35Execution<'_, '_>,
    right_execution: &Qwen35Execution<'_, '_>,
) -> std::result::Result<(), String> {
    assert_eq!(left_execution.position, right_execution.position);
    assert_eq!(left_execution.layers.len(), right_execution.layers.len());
    for (left_layer, right_layer) in left_execution.layers.iter().zip(&right_execution.layers) {
        match (left_layer, right_layer) {
            (LayerState::Recurrent(left), LayerState::Recurrent(right)) => {
                assert_eq!(
                    left.transaction_state_for_test(),
                    right.transaction_state_for_test()
                );
            }
            (LayerState::Full(left_layer), LayerState::Full(right_layer)) => {
                let left_pool = left_execution
                    .paged_kv_pool
                    .as_ref()
                    .ok_or_else(|| "left execution is missing paged KV backing".to_string())?;
                let right_pool = right_execution
                    .paged_kv_pool
                    .as_ref()
                    .ok_or_else(|| "right execution is missing paged KV backing".to_string())?;
                let left_kv = left_pool
                    .layer_kv(*left_layer)
                    .map_err(|error| error.to_string())?;
                let right_kv = right_pool
                    .layer_kv(*right_layer)
                    .map_err(|error| error.to_string())?;
                assert_eq!(left_kv.tokens(), right_kv.tokens());
                for token in 0..left_kv.tokens() {
                    assert_eq!(
                        left_kv.key_row(token).map_err(|error| error.to_string())?,
                        right_kv.key_row(token).map_err(|error| error.to_string())?
                    );
                    assert_eq!(
                        left_kv
                            .value_row(token)
                            .map_err(|error| error.to_string())?,
                        right_kv
                            .value_row(token)
                            .map_err(|error| error.to_string())?
                    );
                }
            }
            _ => return Err("execution plans disagreed on the layer kind".to_string()),
        }
    }
    Ok(())
}
