//! Qwen3.5 GGUF structural preflight.
//!
//! WHY: The tensor-role and cross-field relations are independently derived
//! from `ggml-org/llama.cpp` `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`,
//! `src/models/qwen35.cpp` and `src/llama-hparams.cpp`. This module carries
//! those structural facts into an original, inspection-only boundary; it does
//! not carry upstream implementation code or execution behavior.

use std::collections::{HashMap, HashSet};

use loader::gguf::{MetaValue, MetaValueType, ObservedArtifact, TensorDescriptor};

use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, DuplicateTensorSnafu, MetadataRelationSnafu, MetadataTypeSnafu,
    MissingMetadataSnafu, MissingTensorSnafu, TensorShapeSnafu, UnclassifiedTensorSnafu,
};

const ARCHITECTURE_KEY: &str = "general.architecture";
const ARCHITECTURE_VALUE: &str = "qwen35";
const BLOCK_COUNT_KEY: &str = "qwen35.block_count";
const NEXTN_PREDICT_LAYERS_KEY: &str = "qwen35.nextn_predict_layers";
const FULL_ATTENTION_INTERVAL_KEY: &str = "qwen35.full_attention_interval";
const EMBEDDING_LENGTH_KEY: &str = "qwen35.embedding_length";
const FEED_FORWARD_LENGTH_KEY: &str = "qwen35.feed_forward_length";
const HEAD_COUNT_KEY: &str = "qwen35.attention.head_count";
const KEY_VALUE_HEAD_COUNT_KEY: &str = "qwen35.attention.head_count_kv";
const KEY_LENGTH_KEY: &str = "qwen35.attention.key_length";
const RECURRENT_LAYERS_KEY: &str = "qwen35.attention.recurrent_layers";
const VALUE_LENGTH_KEY: &str = "qwen35.attention.value_length";
const LAYERNORM_RMS_EPSILON_KEY: &str = "qwen35.attention.layer_norm_rms_epsilon";
const SSM_CONV_KERNEL_KEY: &str = "qwen35.ssm.conv_kernel";
const SSM_INNER_SIZE_KEY: &str = "qwen35.ssm.inner_size";
const SSM_STATE_SIZE_KEY: &str = "qwen35.ssm.state_size";
const SSM_TIME_STEP_RANK_KEY: &str = "qwen35.ssm.time_step_rank";
const SSM_GROUP_COUNT_KEY: &str = "qwen35.ssm.group_count";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";

const TOKEN_EMBEDDING_TENSOR: &str = "token_embd.weight";
const OUTPUT_NORM_TENSOR: &str = "output_norm.weight";
const OUTPUT_TENSOR: &str = "output.weight";
const ATTN_GATE_ROLE: &str = "attn_gate.weight";
const ATTN_K_ROLE: &str = "attn_k.weight";
const ATTN_K_NORM_ROLE: &str = "attn_k_norm.weight";
const ATTN_NORM_ROLE: &str = "attn_norm.weight";
const ATTN_OUTPUT_ROLE: &str = "attn_output.weight";
const ATTN_Q_ROLE: &str = "attn_q.weight";
const ATTN_QKV_ROLE: &str = "attn_qkv.weight";
const ATTN_Q_NORM_ROLE: &str = "attn_q_norm.weight";
const ATTN_V_ROLE: &str = "attn_v.weight";
const FFN_DOWN_ROLE: &str = "ffn_down.weight";
const FFN_GATE_ROLE: &str = "ffn_gate.weight";
const FFN_UP_ROLE: &str = "ffn_up.weight";
const NEXTN_EH_PROJ_ROLE: &str = "nextn.eh_proj.weight";
const NEXTN_ENORM_ROLE: &str = "nextn.enorm.weight";
const NEXTN_HNORM_ROLE: &str = "nextn.hnorm.weight";
const NEXTN_SHARED_HEAD_NORM_ROLE: &str = "nextn.shared_head_norm.weight";
const POST_ATTENTION_NORM_ROLE: &str = "post_attention_norm.weight";
const SSM_A_ROLE: &str = "ssm_a";
const SSM_ALPHA_ROLE: &str = "ssm_alpha.weight";
const SSM_BETA_ROLE: &str = "ssm_beta.weight";
const SSM_CONV1D_ROLE: &str = "ssm_conv1d.weight";
const SSM_DT_ROLE: &str = "ssm_dt.bias";
const SSM_NORM_ROLE: &str = "ssm_norm.weight";
const SSM_OUT_ROLE: &str = "ssm_out.weight";

const MAX_NEXTN_LAYERS: u64 = 1;
const Q_PROJECTION_MULTIPLIER: u64 = 2;

/// The sole structural role inventory. Counts and the expected-name/shape map
/// both derive from these templates. Every main block receives the common
/// attention/FFN/norm slice; its kind contributes exactly one specific slice.
/// `NEXTN_EXTENSION_TEMPLATES` deliberately extends the full-attention
/// composition, because a `NextN` block has every full-attention role plus its
/// four terminal roles.
const GLOBAL_TEMPLATES: &[TensorTemplate] = &[
    TensorTemplate::new(TOKEN_EMBEDDING_TENSOR, TemplateShape::HiddenVocabulary),
    TensorTemplate::new(OUTPUT_NORM_TENSOR, TemplateShape::Hidden),
    TensorTemplate::new(OUTPUT_TENSOR, TemplateShape::HiddenVocabulary),
];

const COMMON_BLOCK_TEMPLATES: &[TensorTemplate] = &[
    TensorTemplate::new(ATTN_NORM_ROLE, TemplateShape::Hidden),
    TensorTemplate::new(FFN_DOWN_ROLE, TemplateShape::FeedForwardDown),
    TensorTemplate::new(FFN_GATE_ROLE, TemplateShape::FeedForwardUp),
    TensorTemplate::new(FFN_UP_ROLE, TemplateShape::FeedForwardUp),
    TensorTemplate::new(POST_ATTENTION_NORM_ROLE, TemplateShape::Hidden),
];

const FULL_ATTENTION_TEMPLATES: &[TensorTemplate] = &[
    TensorTemplate::new(ATTN_K_ROLE, TemplateShape::FullAttentionKey),
    TensorTemplate::new(ATTN_K_NORM_ROLE, TemplateShape::KeyWidth),
    TensorTemplate::new(ATTN_OUTPUT_ROLE, TemplateShape::FullAttentionOutput),
    TensorTemplate::new(ATTN_Q_ROLE, TemplateShape::FullAttentionQuery),
    TensorTemplate::new(ATTN_Q_NORM_ROLE, TemplateShape::KeyWidth),
    TensorTemplate::new(ATTN_V_ROLE, TemplateShape::FullAttentionValue),
];

const RECURRENT_TEMPLATES: &[TensorTemplate] = &[
    TensorTemplate::new(ATTN_GATE_ROLE, TemplateShape::RecurrentGate),
    TensorTemplate::new(ATTN_QKV_ROLE, TemplateShape::RecurrentQkv),
    TensorTemplate::new(SSM_A_ROLE, TemplateShape::TimeStepRank),
    TensorTemplate::new(SSM_ALPHA_ROLE, TemplateShape::HiddenTimeStepRank),
    TensorTemplate::new(SSM_BETA_ROLE, TemplateShape::HiddenTimeStepRank),
    TensorTemplate::new(SSM_CONV1D_ROLE, TemplateShape::RecurrentConvolution),
    TensorTemplate::new(SSM_DT_ROLE, TemplateShape::TimeStepRank),
    TensorTemplate::new(SSM_NORM_ROLE, TemplateShape::RecurrentHead),
    TensorTemplate::new(SSM_OUT_ROLE, TemplateShape::RecurrentOutput),
];

const NEXTN_EXTENSION_TEMPLATES: &[TensorTemplate] = &[
    TensorTemplate::new(NEXTN_EH_PROJ_ROLE, TemplateShape::NextNProjection),
    TensorTemplate::new(NEXTN_ENORM_ROLE, TemplateShape::Hidden),
    TensorTemplate::new(NEXTN_HNORM_ROLE, TemplateShape::Hidden),
    TensorTemplate::new(NEXTN_SHARED_HEAD_NORM_ROLE, TemplateShape::Hidden),
];

/// A source-derived Qwen3.5 tensor topology bound to one opaque observation.
///
/// This profile has no constructor from [`loader::gguf::Inspection`], a
/// serialized receipt, or a caller-provided tensor table. It borrows the
/// [`ObservedArtifact`] that performed the retained-file parse and whole-stream
/// digest. A successful preflight establishes only that the observed metadata
/// and descriptors match this narrow structural domain; it is not artifact
/// provenance, model admission, dequantization, execution, or residency proof.
/// GGML storage tags remain loader-validated descriptor facts, not per-role
/// execution admission in this preflight.
/// An explicit `attention.recurrent_layers` override is outside this
/// interval-derived domain and is refused rather than silently ignored.
///
/// ```compile_fail
/// use decoders::Qwen35StructuralProfile;
/// use loader::gguf::Inspection;
///
/// fn forge_from_report(receipt: &Inspection) {
///     let _ = Qwen35StructuralProfile::try_from_observed(receipt);
/// }
/// ```
#[derive(Debug)]
pub struct Qwen35StructuralProfile<'artifact> {
    observed: &'artifact ObservedArtifact,
    dimensions: Dimensions,
}

impl<'artifact> Qwen35StructuralProfile<'artifact> {
    /// Construct a structural profile from an opaque retained-file observation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when required typed metadata, source-derived
    /// relations, tensor names, tensor counts, or tensor shapes do not match the
    /// bounded Qwen3.5 structural domain.
    pub fn try_from_observed(observed: &'artifact ObservedArtifact) -> Result<Self> {
        let dimensions = Dimensions::from_metadata(observed.metadata())?;
        validate_tensor_inventory(&dimensions, observed.tensor_descriptors())?;

        Ok(Self {
            observed,
            dimensions,
        })
    }

    /// Borrow the exact opaque observation that passed this structural preflight.
    #[must_use]
    pub const fn observed(&self) -> &'artifact ObservedArtifact {
        self.observed
    }

    /// Return the number of stored `blk.N` groups, including an optional `NextN` block.
    #[must_use]
    pub const fn stored_block_count(&self) -> u64 {
        self.dimensions.stored_block_count
    }

    /// Return the number of main decoder block groups, excluding `NextN`.
    #[must_use]
    pub const fn main_block_count(&self) -> u64 {
        self.dimensions.main_block_count
    }

    /// Return the number of terminal `NextN` block groups in this narrow domain.
    #[must_use]
    pub const fn nextn_block_count(&self) -> u64 {
        self.dimensions.nextn_block_count
    }

    pub(crate) const fn recurrent_layout(&self) -> Qwen35RecurrentLayout {
        self.dimensions.recurrent_layout()
    }

    pub(crate) const fn execution_dimensions(&self) -> Qwen35ExecutionDimensions {
        self.dimensions.execution_dimensions()
    }
}

#[derive(Debug)]
struct Dimensions {
    hidden: u64,
    feed_forward: u64,
    heads: u64,
    key_value_heads: u64,
    key_width: u64,
    value_width: u64,
    conv_kernel: u64,
    inner: u64,
    state: u64,
    time_step_rank: u64,
    group_count: u64,
    vocabulary: u64,
    stored_block_count: u64,
    main_block_count: u64,
    nextn_block_count: u64,
    full_attention_interval: u64,
}

impl Dimensions {
    fn from_metadata(metadata: &HashMap<String, MetaValue>) -> Result<Self> {
        require_architecture(metadata)?;
        if metadata.contains_key(RECURRENT_LAYERS_KEY) {
            return MetadataRelationSnafu {
                key: RECURRENT_LAYERS_KEY,
                rule: "attention.recurrent_layers is outside the interval-derived structural domain",
            }
            .fail();
        }

        let stored_block_count = required_u32(metadata, BLOCK_COUNT_KEY)?;
        let nextn_block_count = required_u32(metadata, NEXTN_PREDICT_LAYERS_KEY)?;
        if nextn_block_count > MAX_NEXTN_LAYERS {
            return MetadataRelationSnafu {
                key: NEXTN_PREDICT_LAYERS_KEY,
                rule: "the bounded structural domain permits only zero or one NextN block",
            }
            .fail();
        }
        let main_block_count = stored_block_count
            .checked_sub(nextn_block_count)
            .ok_or_else(|| {
                MetadataRelationSnafu {
                    key: BLOCK_COUNT_KEY,
                    rule: "block_count must be at least nextn_predict_layers",
                }
                .build()
            })?;
        if main_block_count == 0 {
            return MetadataRelationSnafu {
                key: BLOCK_COUNT_KEY,
                rule: "at least one main decoder block is required",
            }
            .fail();
        }

        let full_attention_interval = required_u32(metadata, FULL_ATTENTION_INTERVAL_KEY)?;
        if full_attention_interval == 0 {
            return MetadataRelationSnafu {
                key: FULL_ATTENTION_INTERVAL_KEY,
                rule: "full_attention_interval must be non-zero",
            }
            .fail();
        }

        let dimensions = Self {
            hidden: required_u32(metadata, EMBEDDING_LENGTH_KEY)?,
            feed_forward: required_u32(metadata, FEED_FORWARD_LENGTH_KEY)?,
            heads: required_u32(metadata, HEAD_COUNT_KEY)?,
            key_value_heads: required_u32(metadata, KEY_VALUE_HEAD_COUNT_KEY)?,
            key_width: required_u32(metadata, KEY_LENGTH_KEY)?,
            value_width: required_u32(metadata, VALUE_LENGTH_KEY)?,
            conv_kernel: required_u32(metadata, SSM_CONV_KERNEL_KEY)?,
            inner: required_u32(metadata, SSM_INNER_SIZE_KEY)?,
            state: required_u32(metadata, SSM_STATE_SIZE_KEY)?,
            time_step_rank: required_u32(metadata, SSM_TIME_STEP_RANK_KEY)?,
            group_count: required_u32(metadata, SSM_GROUP_COUNT_KEY)?,
            vocabulary: required_vocabulary(metadata)?,
            stored_block_count,
            main_block_count,
            nextn_block_count,
            full_attention_interval,
        };
        dimensions.validate_relations()?;
        Ok(dimensions)
    }

    fn validate_relations(&self) -> Result<()> {
        for (key, value) in [
            (EMBEDDING_LENGTH_KEY, self.hidden),
            (FEED_FORWARD_LENGTH_KEY, self.feed_forward),
            (HEAD_COUNT_KEY, self.heads),
            (KEY_VALUE_HEAD_COUNT_KEY, self.key_value_heads),
            (KEY_LENGTH_KEY, self.key_width),
            (VALUE_LENGTH_KEY, self.value_width),
            (SSM_CONV_KERNEL_KEY, self.conv_kernel),
            (SSM_INNER_SIZE_KEY, self.inner),
            (SSM_STATE_SIZE_KEY, self.state),
            (SSM_TIME_STEP_RANK_KEY, self.time_step_rank),
            (SSM_GROUP_COUNT_KEY, self.group_count),
            (TOKENS_KEY, self.vocabulary),
        ] {
            if value == 0 {
                return MetadataRelationSnafu {
                    key,
                    rule: "must be non-zero",
                }
                .fail();
            }
        }
        if self.key_value_heads > self.heads {
            return MetadataRelationSnafu {
                key: KEY_VALUE_HEAD_COUNT_KEY,
                rule: "attention.head_count_kv must not exceed attention.head_count",
            }
            .fail();
        }
        if !self.heads.is_multiple_of(self.key_value_heads) {
            return MetadataRelationSnafu {
                key: KEY_VALUE_HEAD_COUNT_KEY,
                rule: "attention.head_count must be divisible by attention.head_count_kv",
            }
            .fail();
        }
        if self.key_width != self.value_width {
            return MetadataRelationSnafu {
                key: VALUE_LENGTH_KEY,
                rule: "attention.value_length must equal attention.key_length for Qwen3.5 NextN",
            }
            .fail();
        }
        if !self.inner.is_multiple_of(self.time_step_rank) {
            return MetadataRelationSnafu {
                key: SSM_INNER_SIZE_KEY,
                rule: "ssm.inner_size must be divisible by ssm.time_step_rank",
            }
            .fail();
        }
        if self.inner / self.time_step_rank != self.state {
            return MetadataRelationSnafu {
                key: SSM_STATE_SIZE_KEY,
                rule: "ssm.state_size must equal ssm.inner_size / ssm.time_step_rank",
            }
            .fail();
        }
        if !self.time_step_rank.is_multiple_of(self.group_count) {
            return MetadataRelationSnafu {
                key: SSM_TIME_STEP_RANK_KEY,
                rule: "ssm.time_step_rank must be divisible by ssm.group_count",
            }
            .fail();
        }
        Ok(())
    }

    fn full_attention_count(&self) -> u64 {
        self.main_block_count / self.full_attention_interval
    }

    fn recurrent_count(&self) -> u64 {
        self.main_block_count - self.full_attention_count()
    }

    fn expected_tensor_count(&self) -> Result<u64> {
        let global = template_count(GLOBAL_TEMPLATES, "global tensor inventory")?;
        let full_attention_roles =
            block_template_count(FULL_ATTENTION_TEMPLATES, "full-attention tensor inventory")?;
        let recurrent_roles =
            block_template_count(RECURRENT_TEMPLATES, "recurrent tensor inventory")?;
        let nextn_roles = checked_add(
            full_attention_roles,
            template_count(
                NEXTN_EXTENSION_TEMPLATES,
                "NextN extension tensor inventory",
            )?,
            "NextN tensor inventory",
        )?;
        let recurrent = checked_mul(
            self.recurrent_count(),
            recurrent_roles,
            "recurrent tensor inventory",
        )?;
        let full_attention = checked_mul(
            self.full_attention_count(),
            full_attention_roles,
            "full-attention tensor inventory",
        )?;
        let nextn = checked_mul(
            self.nextn_block_count,
            nextn_roles,
            "NextN tensor inventory",
        )?;
        checked_add(
            checked_add(
                checked_add(global, recurrent, "global plus recurrent tensor inventory")?,
                full_attention,
                "full structural tensor inventory",
            )?,
            nextn,
            "complete structural tensor inventory",
        )
    }

    fn full_attention_block(&self, block_index: u64) -> bool {
        (block_index + 1).is_multiple_of(self.full_attention_interval)
    }

    fn full_attention_shapes(&self) -> Result<FullAttentionShapes> {
        let query_width = checked_mul(
            checked_mul(
                self.heads,
                self.key_width,
                "attention query heads times key width",
            )?,
            Q_PROJECTION_MULTIPLIER,
            "attention query projection multiplier",
        )?;
        let key_width = checked_mul(
            self.key_value_heads,
            self.key_width,
            "attention key-value heads times key width",
        )?;
        let value_width = checked_mul(
            self.key_value_heads,
            self.value_width,
            "attention key-value heads times value width",
        )?;
        let output_width = checked_mul(
            self.heads,
            self.key_width,
            "attention output heads times key width",
        )?;
        Ok(FullAttentionShapes {
            query: query_width,
            key: key_width,
            value: value_width,
            output: output_width,
        })
    }

    fn recurrent_shapes(&self) -> Result<RecurrentShapes> {
        let key_width = checked_mul(
            self.state,
            self.group_count,
            "SSM state size times group count",
        )?;
        let double_key_width =
            checked_mul(key_width, Q_PROJECTION_MULTIPLIER, "double SSM key width")?;
        let conv_width = checked_add(double_key_width, self.inner, "SSM convolution width")?;
        Ok(RecurrentShapes {
            conv_width,
            head_width: self.inner / self.time_step_rank,
        })
    }

    const fn recurrent_layout(&self) -> Qwen35RecurrentLayout {
        Qwen35RecurrentLayout {
            hidden: self.hidden,
            conv_kernel: self.conv_kernel,
            inner: self.inner,
            state: self.state,
            time_step_rank: self.time_step_rank,
            group_count: self.group_count,
            main_block_count: self.main_block_count,
            full_attention_interval: self.full_attention_interval,
        }
    }

    const fn execution_dimensions(&self) -> Qwen35ExecutionDimensions {
        Qwen35ExecutionDimensions {
            hidden: self.hidden,
            feed_forward: self.feed_forward,
            heads: self.heads,
            key_value_heads: self.key_value_heads,
            key_width: self.key_width,
            vocabulary: self.vocabulary,
            main_block_count: self.main_block_count,
            full_attention_interval: self.full_attention_interval,
        }
    }
}

/// Recurrent dimensions retained after the one authoritative metadata parse.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen35RecurrentLayout {
    pub(crate) hidden: u64,
    pub(crate) conv_kernel: u64,
    pub(crate) inner: u64,
    pub(crate) state: u64,
    pub(crate) time_step_rank: u64,
    pub(crate) group_count: u64,
    pub(crate) main_block_count: u64,
    pub(crate) full_attention_interval: u64,
}

/// Checked structural dimensions reused by payload-bound execution.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen35ExecutionDimensions {
    pub(crate) hidden: u64,
    pub(crate) feed_forward: u64,
    pub(crate) heads: u64,
    pub(crate) key_value_heads: u64,
    pub(crate) key_width: u64,
    pub(crate) vocabulary: u64,
    pub(crate) main_block_count: u64,
    pub(crate) full_attention_interval: u64,
}

pub(crate) fn recurrent_layernorm_rms_epsilon(
    metadata: &HashMap<String, MetaValue>,
) -> Result<f32> {
    let epsilon = required_f32(metadata, LAYERNORM_RMS_EPSILON_KEY)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return MetadataRelationSnafu {
            key: LAYERNORM_RMS_EPSILON_KEY,
            rule: "attention.layer_norm_rms_epsilon must be finite and positive for recurrent execution",
        }
        .fail();
    }
    Ok(epsilon)
}

#[derive(Debug)]
struct FullAttentionShapes {
    query: u64,
    key: u64,
    value: u64,
    output: u64,
}

#[derive(Debug)]
struct RecurrentShapes {
    conv_width: u64,
    head_width: u64,
}

#[derive(Clone, Copy)]
struct TensorTemplate {
    role: &'static str,
    shape: TemplateShape,
}

impl TensorTemplate {
    const fn new(role: &'static str, shape: TemplateShape) -> Self {
        Self { role, shape }
    }

    fn shape(self, dimensions: &Dimensions) -> Result<Vec<u64>> {
        match self.shape {
            TemplateShape::HiddenVocabulary => Ok(vec![dimensions.hidden, dimensions.vocabulary]),
            TemplateShape::Hidden => Ok(vec![dimensions.hidden]),
            TemplateShape::KeyWidth => Ok(vec![dimensions.key_width]),
            TemplateShape::FullAttentionKey => {
                let shapes = dimensions.full_attention_shapes()?;
                Ok(vec![dimensions.hidden, shapes.key])
            }
            TemplateShape::FullAttentionOutput => {
                let shapes = dimensions.full_attention_shapes()?;
                Ok(vec![shapes.output, dimensions.hidden])
            }
            TemplateShape::FullAttentionQuery => {
                let shapes = dimensions.full_attention_shapes()?;
                Ok(vec![dimensions.hidden, shapes.query])
            }
            TemplateShape::FullAttentionValue => {
                let shapes = dimensions.full_attention_shapes()?;
                Ok(vec![dimensions.hidden, shapes.value])
            }
            TemplateShape::FeedForwardDown => Ok(vec![dimensions.feed_forward, dimensions.hidden]),
            TemplateShape::FeedForwardUp => Ok(vec![dimensions.hidden, dimensions.feed_forward]),
            TemplateShape::RecurrentGate => Ok(vec![dimensions.hidden, dimensions.inner]),
            TemplateShape::RecurrentQkv => {
                let shapes = dimensions.recurrent_shapes()?;
                Ok(vec![dimensions.hidden, shapes.conv_width])
            }
            TemplateShape::TimeStepRank => Ok(vec![dimensions.time_step_rank]),
            TemplateShape::HiddenTimeStepRank => {
                Ok(vec![dimensions.hidden, dimensions.time_step_rank])
            }
            TemplateShape::RecurrentConvolution => {
                let shapes = dimensions.recurrent_shapes()?;
                Ok(vec![dimensions.conv_kernel, shapes.conv_width])
            }
            TemplateShape::RecurrentHead => {
                let shapes = dimensions.recurrent_shapes()?;
                Ok(vec![shapes.head_width])
            }
            TemplateShape::RecurrentOutput => Ok(vec![dimensions.inner, dimensions.hidden]),
            TemplateShape::NextNProjection => Ok(vec![
                checked_mul(
                    dimensions.hidden,
                    Q_PROJECTION_MULTIPLIER,
                    "NextN hidden projection width",
                )?,
                dimensions.hidden,
            ]),
        }
    }
}

#[derive(Clone, Copy)]
enum TemplateShape {
    HiddenVocabulary,
    Hidden,
    KeyWidth,
    FullAttentionKey,
    FullAttentionOutput,
    FullAttentionQuery,
    FullAttentionValue,
    FeedForwardDown,
    FeedForwardUp,
    RecurrentGate,
    RecurrentQkv,
    TimeStepRank,
    HiddenTimeStepRank,
    RecurrentConvolution,
    RecurrentHead,
    RecurrentOutput,
    NextNProjection,
}

fn template_count(templates: &[TensorTemplate], context: &'static str) -> Result<u64> {
    u64::try_from(templates.len()).map_err(|_| ArithmeticOverflowSnafu { context }.build())
}

fn block_template_count(
    specific_templates: &[TensorTemplate],
    context: &'static str,
) -> Result<u64> {
    checked_add(
        template_count(COMMON_BLOCK_TEMPLATES, context)?,
        template_count(specific_templates, context)?,
        context,
    )
}

fn validate_tensor_inventory(dimensions: &Dimensions, tensors: &[TensorDescriptor]) -> Result<()> {
    let expected_count = dimensions.expected_tensor_count()?;
    let actual_count = u64::try_from(tensors.len()).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "observed tensor descriptor count",
        }
        .build()
    })?;
    if actual_count != expected_count {
        return MetadataRelationSnafu {
            key: BLOCK_COUNT_KEY,
            rule: "tensor count must derive exactly from the structural role inventory",
        }
        .fail();
    }

    let mut expected = expected_tensors(dimensions)?;
    let mut seen = HashSet::with_capacity(tensors.len());
    for tensor in tensors {
        if !seen.insert(tensor.name.as_str()) {
            return DuplicateTensorSnafu {
                name: tensor.name.clone(),
            }
            .fail();
        }
        let Some(shape) = expected.remove(tensor.name.as_str()) else {
            return UnclassifiedTensorSnafu {
                name: tensor.name.clone(),
            }
            .fail();
        };
        if tensor.dims != shape {
            return TensorShapeSnafu {
                name: tensor.name.clone(),
                expected: shape,
                actual: tensor.dims.clone(),
            }
            .fail();
        }
    }

    let Some((name, _)) = expected.into_iter().next() else {
        return Ok(());
    };
    MissingTensorSnafu { name }.fail()
}

fn expected_tensors(dimensions: &Dimensions) -> Result<HashMap<String, Vec<u64>>> {
    let expected_count = dimensions.expected_tensor_count()?;
    let capacity = usize::try_from(expected_count).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "expected tensor inventory capacity",
        }
        .build()
    })?;
    let mut tensors = HashMap::with_capacity(capacity);
    add_templates(&mut tensors, dimensions, None, GLOBAL_TEMPLATES)?;

    for block_index in 0..dimensions.main_block_count {
        if dimensions.full_attention_block(block_index) {
            add_block_templates(
                &mut tensors,
                dimensions,
                block_index,
                FULL_ATTENTION_TEMPLATES,
            )?;
        } else {
            add_block_templates(&mut tensors, dimensions, block_index, RECURRENT_TEMPLATES)?;
        }
    }
    for nextn_offset in 0..dimensions.nextn_block_count {
        let block_index = checked_add(
            dimensions.main_block_count,
            nextn_offset,
            "NextN block index",
        )?;
        add_block_templates(
            &mut tensors,
            dimensions,
            block_index,
            FULL_ATTENTION_TEMPLATES,
        )?;
        add_templates(
            &mut tensors,
            dimensions,
            Some(block_index),
            NEXTN_EXTENSION_TEMPLATES,
        )?;
    }
    Ok(tensors)
}

fn add_block_templates(
    tensors: &mut HashMap<String, Vec<u64>>,
    dimensions: &Dimensions,
    block_index: u64,
    specific_templates: &[TensorTemplate],
) -> Result<()> {
    add_templates(
        tensors,
        dimensions,
        Some(block_index),
        COMMON_BLOCK_TEMPLATES,
    )?;
    add_templates(tensors, dimensions, Some(block_index), specific_templates)
}

fn add_templates(
    tensors: &mut HashMap<String, Vec<u64>>,
    dimensions: &Dimensions,
    block_index: Option<u64>,
    templates: &[TensorTemplate],
) -> Result<()> {
    for template in templates {
        let name = match block_index {
            Some(index) => block_tensor_name(index, template.role),
            None => template.role.to_string(),
        };
        add_tensor(tensors, name, template.shape(dimensions)?);
    }
    Ok(())
}

fn add_tensor(tensors: &mut HashMap<String, Vec<u64>>, name: String, shape: Vec<u64>) {
    let replaced = tensors.insert(name, shape);
    debug_assert!(replaced.is_none(), "structural role names must be unique");
}

fn block_tensor_name(block_index: u64, role: &str) -> String {
    format!("blk.{block_index}.{role}")
}

fn require_architecture(metadata: &HashMap<String, MetaValue>) -> Result<()> {
    let Some(value) = metadata.get(ARCHITECTURE_KEY) else {
        return MissingMetadataSnafu {
            key: ARCHITECTURE_KEY,
        }
        .fail();
    };
    let MetaValue::String(architecture) = value else {
        return MetadataTypeSnafu {
            key: ARCHITECTURE_KEY,
            expected: "string",
            actual: value.value_type(),
        }
        .fail();
    };
    if architecture != ARCHITECTURE_VALUE {
        return MetadataRelationSnafu {
            key: ARCHITECTURE_KEY,
            rule: "general.architecture must be qwen35",
        }
        .fail();
    }
    Ok(())
}

fn required_u32(metadata: &HashMap<String, MetaValue>, key: &'static str) -> Result<u64> {
    let Some(value) = metadata.get(key) else {
        return MissingMetadataSnafu { key }.fail();
    };
    let MetaValue::U32(value) = value else {
        return MetadataTypeSnafu {
            key,
            expected: "u32",
            actual: value.value_type(),
        }
        .fail();
    };
    Ok(u64::from(*value))
}

fn required_f32(metadata: &HashMap<String, MetaValue>, key: &'static str) -> Result<f32> {
    let Some(value) = metadata.get(key) else {
        return MissingMetadataSnafu { key }.fail();
    };
    let MetaValue::F32(value) = value else {
        return MetadataTypeSnafu {
            key,
            expected: "f32",
            actual: value.value_type(),
        }
        .fail();
    };
    Ok(*value)
}

fn required_vocabulary(metadata: &HashMap<String, MetaValue>) -> Result<u64> {
    let Some(value) = metadata.get(TOKENS_KEY) else {
        return MissingMetadataSnafu { key: TOKENS_KEY }.fail();
    };
    let MetaValue::Array(tokens) = value else {
        return MetadataTypeSnafu {
            key: TOKENS_KEY,
            expected: "array<string>",
            actual: value.value_type(),
        }
        .fail();
    };
    if tokens.element_type() != MetaValueType::String {
        return MetadataRelationSnafu {
            key: TOKENS_KEY,
            rule: "tokenizer.ggml.tokens must declare string array elements",
        }
        .fail();
    }
    for token in tokens.values() {
        if token.value_type() != MetaValueType::String {
            return MetadataRelationSnafu {
                key: TOKENS_KEY,
                rule: "tokenizer.ggml.tokens values must all be strings",
            }
            .fail();
        }
    }
    u64::try_from(tokens.values().len()).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "tokenizer vocabulary length",
        }
        .build()
    })
}

fn checked_mul(left: u64, right: u64, context: &'static str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn checked_add(left: u64, right: u64, context: &'static str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

#[cfg(test)]
#[path = "qwen35_tests.rs"]
mod tests;
