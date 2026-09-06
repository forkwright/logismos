//! Digest-verified CPU recurrent-attention execution for one Qwen3.5 block.
//!
//! WHY: the structural profile recognizes a Qwen3.5 descriptor inventory but
//! deliberately does not execute it. This module binds one recurrent layer's
//! finite payload parameters and persistent state to the verified artifact.

use kernels::{
    CausalConvInput, MultiHeadRecurrentInput, causal_conv_fwd, multi_head_recurrent_fwd,
};
use loader::gguf::GgmlType;
use quant::f32_row::F32Row;
use snafu::ResultExt;

use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, PayloadTensorSnafu, ProjectionBytesSnafu, ProjectionDtypeSnafu,
    ProjectionRowSnafu, RecurrentAllocationSnafu, RecurrentArithmeticSnafu,
    RecurrentConvolutionSnafu, RecurrentGdnSnafu, RecurrentInputSnafu, RecurrentLayerSnafu,
    RecurrentRmsNormSnafu, TensorShapeSnafu,
};
use crate::qwen35::{Qwen35RecurrentLayout, recurrent_layernorm_rms_epsilon};
use crate::qwen35_weights::Qwen35Weights;

const ATTN_GATE_ROLE: &str = "attn_gate.weight";
const ATTN_NORM_ROLE: &str = "attn_norm.weight";
const ATTN_QKV_ROLE: &str = "attn_qkv.weight";
const SSM_A_ROLE: &str = "ssm_a";
const SSM_ALPHA_ROLE: &str = "ssm_alpha.weight";
const SSM_BETA_ROLE: &str = "ssm_beta.weight";
const SSM_CONV1D_ROLE: &str = "ssm_conv1d.weight";
const SSM_DT_ROLE: &str = "ssm_dt.bias";
const SSM_NORM_ROLE: &str = "ssm_norm.weight";
const SSM_OUT_ROLE: &str = "ssm_out.weight";

/// Stateful CPU execution for one digest-verified Qwen3.5 recurrent block.
///
/// This is a narrow recurrent-attention trunk: it performs attention `RMSNorm`,
/// QKV/Z/alpha/beta projections, causal convolution, Q/K L2 normalization,
/// delta recurrence, gated `RMSNorm`, and output projection. It does not execute
/// residuals, FFN, dense attention, `NextN`, tokenization, logits, or a model.
#[derive(Debug)]
pub struct Qwen35RecurrentExecution<'weights, 'artifact> {
    weights: &'weights Qwen35Weights<'artifact>,
    block_index: u64,
    layout: ExecutionLayout,
    attention_norm: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_conv: Vec<f32>,
    ssm_dt: Vec<f32>,
    ssm_norm: Vec<f32>,
    convolution_history: Vec<f32>,
    recurrent_state: Vec<f32>,
}

impl<'weights, 'artifact> Qwen35RecurrentExecution<'weights, 'artifact> {
    pub(crate) fn try_from_weights(
        weights: &'weights Qwen35Weights<'artifact>,
        block_index: u64,
    ) -> Result<Self> {
        let epsilon = recurrent_layernorm_rms_epsilon(weights.payload().observation().metadata())?;
        let layout = ExecutionLayout::try_from_profile(weights.recurrent_layout(), epsilon)?;
        layout.validate_recurrent_block(block_index)?;
        let attention_norm = read_f32_tensor(
            weights,
            &block_tensor_name(block_index, ATTN_NORM_ROLE),
            &[layout.hidden_u64],
        )?;
        let ssm_a = read_f32_tensor(
            weights,
            &block_tensor_name(block_index, SSM_A_ROLE),
            &[layout.value_head_count_u64],
        )?;
        let ssm_dt = read_f32_tensor(
            weights,
            &block_tensor_name(block_index, SSM_DT_ROLE),
            &[layout.value_head_count_u64],
        )?;
        let ssm_norm = read_f32_tensor(
            weights,
            &block_tensor_name(block_index, SSM_NORM_ROLE),
            &[layout.value_dim_u64],
        )?;
        let ssm_conv = read_f32_tensor(
            weights,
            &block_tensor_name(block_index, SSM_CONV1D_ROLE),
            &[layout.conv_kernel_u64, layout.conv_width_u64],
        )?;
        let convolution_history = zeroed_f32(
            "convolution history",
            checked_product(
                layout.conv_width,
                layout.conv_kernel.checked_sub(1).ok_or_else(|| {
                    ArithmeticOverflowSnafu {
                        context: "recurrent convolution kernel minus one",
                    }
                    .build()
                })?,
                "recurrent convolution history",
            )?,
        )?;
        let recurrent_state = zeroed_f32(
            "GDN state",
            checked_product(
                layout.value_head_count,
                checked_product(layout.key_dim, layout.value_dim, "recurrent GDN state head")?,
                "recurrent GDN state",
            )?,
        )?;

        Ok(Self {
            weights,
            block_index,
            layout,
            attention_norm,
            ssm_a,
            ssm_conv,
            ssm_dt,
            ssm_norm,
            convolution_history,
            recurrent_state,
        })
    }

    /// Execute one or more complete hidden-width tokens transactionally.
    ///
    /// On error, this object retains its exact prior convolution and recurrent
    /// state; no partial output or state update escapes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when input, payload rows, finite arithmetic, or
    /// one of the bounded CPU reference operators rejects the step.
    pub fn step(&mut self, hidden_tokens: &[f32]) -> Result<Vec<f32>> {
        let (token_count, normalized) = self.normalize_input(hidden_tokens)?;
        let projected = self.project_inputs(token_count, &normalized)?;
        let recurrent = self.run_recurrence(token_count, projected)?;
        let output = self.project_output(token_count, &recurrent.output, &recurrent.z)?;

        self.convolution_history = recurrent.convolution_history;
        self.recurrent_state = recurrent.state;
        Ok(output)
    }

    #[cfg(test)]
    pub(crate) fn state_for_test(&self) -> &[f32] {
        &self.recurrent_state
    }

    fn normalize_input(&self, hidden_tokens: &[f32]) -> Result<(usize, Vec<f32>)> {
        if hidden_tokens.is_empty() || !hidden_tokens.len().is_multiple_of(self.layout.hidden) {
            return RecurrentInputSnafu {
                hidden: self.layout.hidden,
                actual: hidden_tokens.len(),
            }
            .fail();
        }
        ensure_finite(hidden_tokens, "input", 0)?;
        let token_count = hidden_tokens.len() / self.layout.hidden;
        let normalized = kernels::cpu_f32::rms_norm(
            hidden_tokens,
            &self.attention_norm,
            token_count,
            self.layout.hidden,
            self.layout.epsilon,
        )
        .context(RecurrentRmsNormSnafu)?;
        ensure_finite(&normalized, "attention RMSNorm", 0)?;
        Ok((token_count, normalized))
    }

    fn project_inputs(&self, token_count: usize, normalized: &[f32]) -> Result<ProjectedInputs> {
        let qkv = project_tokens(
            self.weights,
            &block_tensor_name(self.block_index, ATTN_QKV_ROLE),
            normalized,
            token_count,
            self.layout.hidden,
            self.layout.conv_width,
        )?;
        let z = project_tokens(
            self.weights,
            &block_tensor_name(self.block_index, ATTN_GATE_ROLE),
            normalized,
            token_count,
            self.layout.hidden,
            self.layout.inner,
        )?;
        let alpha = project_tokens(
            self.weights,
            &block_tensor_name(self.block_index, SSM_ALPHA_ROLE),
            normalized,
            token_count,
            self.layout.hidden,
            self.layout.value_head_count,
        )?;
        let beta_projection = project_tokens(
            self.weights,
            &block_tensor_name(self.block_index, SSM_BETA_ROLE),
            normalized,
            token_count,
            self.layout.hidden,
            self.layout.value_head_count,
        )?;
        Ok(ProjectedInputs {
            qkv,
            z,
            alpha,
            beta_projection,
        })
    }

    fn run_recurrence(
        &self,
        token_count: usize,
        projected: ProjectedInputs,
    ) -> Result<RecurrentOutput> {
        let scalars = self.recurrence_scalars(token_count, &projected)?;
        let convolution = self.convolve(token_count, &projected.qkv)?;
        let inputs = self.arrange_recurrence(token_count, convolution.output())?;
        let recurrence = MultiHeadRecurrentInput::new(
            &inputs.q,
            &inputs.k,
            &inputs.v,
            &scalars.beta,
            &scalars.gate,
            self.layout.gdn_scale,
            &self.recurrent_state,
            token_count,
            self.layout.value_head_count,
            self.layout.value_head_count,
            self.layout.key_dim,
            self.layout.value_dim,
        )
        .context(RecurrentGdnSnafu)?;
        let recurrence = multi_head_recurrent_fwd(&recurrence).context(RecurrentGdnSnafu)?;
        Ok(RecurrentOutput {
            output: heads_to_tokens(
                recurrence.output(),
                token_count,
                self.layout.value_head_count,
                self.layout.value_dim,
            )?,
            z: projected.z,
            convolution_history: convolution.history().to_vec(),
            state: recurrence.state().to_vec(),
        })
    }

    fn recurrence_scalars(
        &self,
        token_count: usize,
        projected: &ProjectedInputs,
    ) -> Result<RecurrenceScalars> {
        let beta_tokens = sigmoid(&projected.beta_projection)?;
        let beta = heads_from_tokens(
            &beta_tokens,
            token_count,
            self.layout.value_head_count,
            0,
            self.layout.value_head_count,
            1,
        )?;
        let gate_tokens = recurrent_gate(
            &projected.alpha,
            &self.ssm_dt,
            &self.ssm_a,
            token_count,
            self.layout.value_head_count,
        )?;
        let gate = heads_from_tokens(
            &gate_tokens,
            token_count,
            self.layout.value_head_count,
            0,
            self.layout.value_head_count,
            1,
        )?;
        Ok(RecurrenceScalars { beta, gate })
    }

    fn convolve(&self, token_count: usize, qkv: &[f32]) -> Result<kernels::CausalConvOutput> {
        let convolution = CausalConvInput::new(
            qkv,
            &self.ssm_conv,
            &self.convolution_history,
            token_count,
            self.layout.conv_width,
            self.layout.conv_kernel,
        )
        .context(RecurrentConvolutionSnafu)?;
        causal_conv_fwd(&convolution).context(RecurrentConvolutionSnafu)
    }

    fn arrange_recurrence(
        &self,
        token_count: usize,
        convolution: &[f32],
    ) -> Result<ArrangedRecurrence> {
        let convolved = kernels::cpu_f32::silu(convolution);
        ensure_finite(&convolved, "convolution SiLU", 0)?;
        let q = l2_heads(&convolved, token_count, 0, self.layout)?;
        let k = l2_heads(&convolved, token_count, self.layout.key_width, self.layout)?;
        let v = heads_from_tokens(
            &convolved,
            token_count,
            self.layout.conv_width,
            self.layout.key_width.checked_mul(2).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "recurrent V channel offset",
                }
                .build()
            })?,
            self.layout.value_head_count,
            self.layout.value_dim,
        )?;
        let q_tiled = tile_key_heads(
            &q,
            token_count,
            self.layout.key_head_count,
            self.layout.value_head_count,
            self.layout.key_dim,
        )?;
        let k_tiled = tile_key_heads(
            &k,
            token_count,
            self.layout.key_head_count,
            self.layout.value_head_count,
            self.layout.key_dim,
        )?;
        Ok(ArrangedRecurrence {
            q: q_tiled,
            k: k_tiled,
            v,
        })
    }

    fn project_output(
        &self,
        token_count: usize,
        recurrence_tokens: &[f32],
        z: &[f32],
    ) -> Result<Vec<f32>> {
        let output_rows = checked_product(
            token_count,
            self.layout.value_head_count,
            "recurrent output RMSNorm rows",
        )?;
        let normalized_output = kernels::cpu_f32::rms_norm(
            recurrence_tokens,
            &self.ssm_norm,
            output_rows,
            self.layout.value_dim,
            self.layout.epsilon,
        )
        .context(RecurrentRmsNormSnafu)?;
        ensure_finite(&normalized_output, "recurrent RMSNorm", 0)?;
        let gated_output =
            kernels::cpu_f32::hadamard(&normalized_output, &kernels::cpu_f32::silu(z));
        ensure_finite(&gated_output, "recurrent output gate", 0)?;
        project_tokens(
            self.weights,
            &block_tensor_name(self.block_index, SSM_OUT_ROLE),
            &gated_output,
            token_count,
            self.layout.inner,
            self.layout.hidden,
        )
    }
}

#[derive(Debug)]
struct ProjectedInputs {
    qkv: Vec<f32>,
    z: Vec<f32>,
    alpha: Vec<f32>,
    beta_projection: Vec<f32>,
}

#[derive(Debug)]
struct RecurrentOutput {
    output: Vec<f32>,
    z: Vec<f32>,
    convolution_history: Vec<f32>,
    state: Vec<f32>,
}

#[derive(Debug)]
struct RecurrenceScalars {
    beta: Vec<f32>,
    gate: Vec<f32>,
}

#[derive(Debug)]
struct ArrangedRecurrence {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct ExecutionLayout {
    hidden: usize,
    hidden_u64: u64,
    conv_kernel: usize,
    conv_kernel_u64: u64,
    inner: usize,
    key_dim: usize,
    value_dim: usize,
    value_dim_u64: u64,
    key_head_count: usize,
    value_head_count: usize,
    value_head_count_u64: u64,
    key_width: usize,
    conv_width: usize,
    conv_width_u64: u64,
    main_block_count: u64,
    full_attention_interval: u64,
    epsilon: f32,
    gdn_scale: f32,
}

impl ExecutionLayout {
    fn try_from_profile(layout: Qwen35RecurrentLayout, epsilon: f32) -> Result<Self> {
        let hidden = usize_dimension(layout.hidden, "recurrent hidden width")?;
        let conv_kernel = usize_dimension(layout.conv_kernel, "recurrent convolution kernel")?;
        let inner = usize_dimension(layout.inner, "recurrent inner width")?;
        let key_dim = usize_dimension(layout.state, "recurrent state width")?;
        let value_head_count =
            usize_dimension(layout.time_step_rank, "recurrent value head count")?;
        let key_head_count = usize_dimension(layout.group_count, "recurrent key head count")?;
        let value_dim = inner.checked_div(value_head_count).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "recurrent value width",
            }
            .build()
        })?;
        let key_width = checked_product(key_dim, key_head_count, "recurrent key width")?;
        let conv_width = checked_product(key_width, 2, "recurrent convolution Q/K width")?
            .checked_add(inner)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "recurrent convolution width",
                }
                .build()
            })?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "the CPU recurrence's f32 scale intentionally matches its f32 inputs"
        )]
        let gdn_scale = (key_dim as f32).sqrt().recip();
        Ok(Self {
            hidden,
            hidden_u64: layout.hidden,
            conv_kernel,
            conv_kernel_u64: layout.conv_kernel,
            inner,
            key_dim,
            value_dim,
            value_dim_u64: u64::try_from(value_dim).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "recurrent value width metadata representation",
                }
                .build()
            })?,
            key_head_count,
            value_head_count,
            value_head_count_u64: layout.time_step_rank,
            key_width,
            conv_width,
            conv_width_u64: u64::try_from(conv_width).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "recurrent convolution width metadata representation",
                }
                .build()
            })?,
            main_block_count: layout.main_block_count,
            full_attention_interval: layout.full_attention_interval,
            epsilon,
            gdn_scale,
        })
    }

    fn validate_recurrent_block(self, block_index: u64) -> Result<()> {
        if block_index >= self.main_block_count {
            return RecurrentLayerSnafu {
                block_index,
                rule: "block index must name a main decoder block",
            }
            .fail();
        }
        if (block_index + 1).is_multiple_of(self.full_attention_interval) {
            return RecurrentLayerSnafu {
                block_index,
                rule: "full-attention cadence blocks do not use the recurrent path",
            }
            .fail();
        }
        Ok(())
    }
}

fn read_f32_tensor(
    weights: &Qwen35Weights<'_>,
    name: &str,
    expected_dims: &[u64],
) -> Result<Vec<f32>> {
    let tensor = weights.payload().tensor(name).context(PayloadTensorSnafu {
        name: name.to_string(),
    })?;
    if tensor.ggml_type() != GgmlType::F32 {
        return ProjectionDtypeSnafu {
            name: tensor.name().to_string(),
            actual: tensor.ggml_type(),
        }
        .fail();
    }
    if tensor.dims() != expected_dims {
        return TensorShapeSnafu {
            name: tensor.name().to_string(),
            expected: expected_dims.to_vec(),
            actual: tensor.dims().to_vec(),
        }
        .fail();
    }
    let expected_values = product_dims(expected_dims)?;
    let row = F32Row::parse(tensor.bytes()).with_context(|_| ProjectionRowSnafu {
        name: tensor.name().to_string(),
        row: 0_usize,
    })?;
    if row.len() != expected_values {
        return ProjectionBytesSnafu {
            name: tensor.name().to_string(),
            expected: expected_values.checked_mul(4).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "recurrent F32 parameter bytes",
                }
                .build()
            })?,
            actual: tensor.bytes().len(),
        }
        .fail();
    }
    let mut values = reserve_f32("F32 parameter", expected_values)?;
    for value_index in 0..expected_values {
        let Some(value) = row.value(value_index) else {
            return ProjectionBytesSnafu {
                name: tensor.name().to_string(),
                expected: expected_values.checked_mul(4).ok_or_else(|| {
                    ArithmeticOverflowSnafu {
                        context: "recurrent F32 parameter bytes",
                    }
                    .build()
                })?,
                actual: tensor.bytes().len(),
            }
            .fail();
        };
        values.push(value);
    }
    Ok(values)
}

fn l2_heads(
    values: &[f32],
    token_count: usize,
    channel_offset: usize,
    layout: ExecutionLayout,
) -> Result<Vec<f32>> {
    let heads = heads_from_tokens(
        values,
        token_count,
        layout.conv_width,
        channel_offset,
        layout.key_head_count,
        layout.key_dim,
    )?;
    let mut normalized = reserve_f32("L2-normalized Q/K", heads.len())?;
    for head_tokens in heads.chunks_exact(layout.key_dim) {
        let mut sum_squares = 0.0_f32;
        for value in head_tokens {
            sum_squares += value * value;
            ensure_finite_scalar(sum_squares, "Q/K L2 sum", normalized.len())?;
        }
        let denominator = sum_squares.sqrt().max(layout.epsilon);
        ensure_finite_scalar(denominator, "Q/K L2 denominator", normalized.len())?;
        for value in head_tokens {
            let normalized_value = value / denominator;
            ensure_finite_scalar(normalized_value, "Q/K L2 output", normalized.len())?;
            normalized.push(normalized_value);
        }
    }
    Ok(normalized)
}

fn heads_from_tokens(
    values: &[f32],
    token_count: usize,
    channel_count: usize,
    channel_offset: usize,
    head_count: usize,
    head_width: usize,
) -> Result<Vec<f32>> {
    let output_len = checked_product(
        checked_product(token_count, head_count, "recurrent head tokens")?,
        head_width,
        "recurrent head values",
    )?;
    let mut output = reserve_f32("head-major recurrence input", output_len)?;
    for head_index in 0..head_count {
        let head_offset = checked_add(
            channel_offset,
            checked_product(head_index, head_width, "recurrent head channel offset")?,
            "recurrent head start",
        )?;
        let head_end = checked_add(head_offset, head_width, "recurrent head end")?;
        for token in values.chunks_exact(channel_count).take(token_count) {
            let Some(head) = token.get(head_offset..head_end) else {
                return RecurrentInputSnafu {
                    hidden: channel_count,
                    actual: token.len(),
                }
                .fail();
            };
            output.extend_from_slice(head);
        }
    }
    Ok(output)
}

fn tile_key_heads(
    grouped: &[f32],
    token_count: usize,
    key_head_count: usize,
    value_head_count: usize,
    key_dim: usize,
) -> Result<Vec<f32>> {
    let output_len = checked_product(
        checked_product(value_head_count, token_count, "tiled Q/K heads")?,
        key_dim,
        "tiled Q/K values",
    )?;
    let mut tiled = reserve_f32("tiled Q/K heads", output_len)?;
    let source_head_len = checked_product(token_count, key_dim, "source Q/K head")?;
    for value_head in 0..value_head_count {
        let source_head = value_head % key_head_count;
        let start = checked_product(source_head, source_head_len, "source Q/K head offset")?;
        let end = checked_add(start, source_head_len, "source Q/K head end")?;
        let Some(source) = grouped.get(start..end) else {
            return RecurrentInputSnafu {
                hidden: source_head_len,
                actual: grouped.len(),
            }
            .fail();
        };
        tiled.extend_from_slice(source);
    }
    Ok(tiled)
}

fn heads_to_tokens(
    values: &[f32],
    token_count: usize,
    head_count: usize,
    head_width: usize,
) -> Result<Vec<f32>> {
    let output_len = checked_product(
        checked_product(token_count, head_count, "recurrent output token heads")?,
        head_width,
        "recurrent output token values",
    )?;
    let mut output = reserve_f32("token-major recurrent output", output_len)?;
    let head_len = checked_product(token_count, head_width, "recurrent output head")?;
    for token_index in 0..token_count {
        for head_index in 0..head_count {
            let start = checked_add(
                checked_product(head_index, head_len, "recurrent output head offset")?,
                checked_product(token_index, head_width, "recurrent output token offset")?,
                "recurrent output row offset",
            )?;
            let end = checked_add(start, head_width, "recurrent output row end")?;
            let Some(row) = values.get(start..end) else {
                return RecurrentInputSnafu {
                    hidden: head_width,
                    actual: values.len(),
                }
                .fail();
            };
            output.extend_from_slice(row);
        }
    }
    Ok(output)
}

fn recurrent_gate(
    alpha: &[f32],
    dt: &[f32],
    a: &[f32],
    token_count: usize,
    value_head_count: usize,
) -> Result<Vec<f32>> {
    let mut output = reserve_f32(
        "recurrent log decay",
        checked_product(token_count, value_head_count, "recurrent log decay")?,
    )?;
    for alpha_row in alpha.chunks_exact(value_head_count).take(token_count) {
        for (head_index, alpha_value) in alpha_row.iter().copied().enumerate() {
            let dt_value = dt.get(head_index).copied().ok_or_else(|| {
                RecurrentInputSnafu {
                    hidden: value_head_count,
                    actual: dt.len(),
                }
                .build()
            })?;
            let a_value = a.get(head_index).copied().ok_or_else(|| {
                RecurrentInputSnafu {
                    hidden: value_head_count,
                    actual: a.len(),
                }
                .build()
            })?;
            let alpha_plus_dt = alpha_value + dt_value;
            ensure_finite_scalar(alpha_plus_dt, "recurrent log-decay input", output.len())?;
            let gate = a_value * softplus(alpha_plus_dt);
            ensure_finite_scalar(gate, "recurrent log decay", output.len())?;
            output.push(gate);
        }
    }
    Ok(output)
}

fn sigmoid(values: &[f32]) -> Result<Vec<f32>> {
    let mut output = reserve_f32("sigmoid beta", values.len())?;
    for value in values {
        let sigmoid = if *value >= 0.0 {
            1.0 / (1.0 + (-*value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        };
        ensure_finite_scalar(sigmoid, "sigmoid beta", output.len())?;
        output.push(sigmoid);
    }
    Ok(output)
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

fn project_tokens(
    weights: &Qwen35Weights<'_>,
    name: &str,
    values: &[f32],
    token_count: usize,
    input_width: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    let output_len = checked_product(token_count, output_width, "recurrent projected output")?;
    let mut output = reserve_f32("recurrent projected output", output_len)?;
    for token in values.chunks_exact(input_width).take(token_count) {
        output.extend(weights.project(name, token)?);
    }
    Ok(output)
}

fn product_dims(dimensions: &[u64]) -> Result<usize> {
    dimensions
        .iter()
        .copied()
        .try_fold(1_usize, |product, dimension| {
            checked_product(
                product,
                usize_dimension(dimension, "recurrent parameter dimension")?,
                "recurrent parameter values",
            )
        })
}

fn zeroed_f32(target: &'static str, length: usize) -> Result<Vec<f32>> {
    let mut values = reserve_f32(target, length)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn reserve_f32(target: &'static str, length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .with_context(|_| RecurrentAllocationSnafu { target, length })?;
    Ok(values)
}

fn ensure_finite(values: &[f32], stage: &'static str, offset: usize) -> Result<()> {
    for (index, value) in values.iter().copied().enumerate() {
        ensure_finite_scalar(
            value,
            stage,
            checked_add(offset, index, "recurrent finite index")?,
        )?;
    }
    Ok(())
}

fn ensure_finite_scalar(value: f32, stage: &'static str, index: usize) -> Result<()> {
    if !value.is_finite() {
        return RecurrentArithmeticSnafu { stage, index }.fail();
    }
    Ok(())
}

fn checked_product(left: usize, right: usize, context: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn checked_add(left: usize, right: usize, context: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn usize_dimension(value: u64, context: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| ArithmeticOverflowSnafu { context }.build())
}

fn block_tensor_name(block_index: u64, role: &str) -> String {
    format!("blk.{block_index}.{role}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_qk_heads_in_direct_gguf_tiled_order() -> Result<()> {
        let key_head_count = 2;
        let value_head_count = 4;
        let key_dim = 2;
        let grouped_q = [1.0_f32, 2.0, 10.0, 20.0];

        let tiled = tile_key_heads(&grouped_q, 1, key_head_count, value_head_count, key_dim)?;
        let contiguous_grouped = [1.0_f32, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0];

        assert_eq!(
            tiled,
            vec![1.0, 2.0, 10.0, 20.0, 1.0, 2.0, 10.0, 20.0],
            "GGUF tiled V-head order maps each value head to key head hv % Hk"
        );
        assert_ne!(
            tiled, contiguous_grouped,
            "contiguous generic grouping would pair the middle value heads with the wrong Q/K head"
        );
        Ok(())
    }

    #[test]
    fn applies_qwen_l2_epsilon_as_a_denominator_clamp() -> Result<()> {
        let values = [3.0_f32, 4.0];
        let epsilon = 6.0_f32;

        let layout = ExecutionLayout {
            hidden: 1,
            hidden_u64: 1,
            conv_kernel: 1,
            conv_kernel_u64: 1,
            inner: 1,
            key_dim: 2,
            value_dim: 1,
            value_dim_u64: 1,
            key_head_count: 1,
            value_head_count: 1,
            value_head_count_u64: 1,
            key_width: 2,
            conv_width: 2,
            conv_width_u64: 2,
            main_block_count: 1,
            full_attention_interval: 2,
            epsilon,
            gdn_scale: 1.0,
        };
        let normalized = l2_heads(&values, 1, 0, layout)?;
        let oracle = [
            f64::from(values[0]) / f64::from(epsilon),
            f64::from(values[1]) / f64::from(epsilon),
        ];

        for (actual, expected) in normalized.iter().zip(oracle) {
            assert!(
                (f64::from(*actual) - expected).abs() < 1.0e-6,
                "Qwen L2 must use max(sqrt(sum_sq), epsilon), not sqrt(sum_sq + epsilon)"
            );
        }
        Ok(())
    }
}
