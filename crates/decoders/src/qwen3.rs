//! Bounded native CPU Qwen3 causal execution with strict embedding and rank profiles.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use loader::gguf::{GgmlType, MetaValue, MetaValueType, VerifiedArtifact};
use num_traits::ToPrimitive;
use snafu::ResultExt;

use crate::Result;
use crate::error::{
    Qwen3AllocationSnafu, Qwen3ArithmeticSnafu, Qwen3CpuSnafu, Qwen3ExecutionSnafu,
    Qwen3MetadataSnafu, Qwen3TensorSnafu,
};
use crate::matrix::CheckedMatrix;
use crate::qwen3_requirements::{Qwen3AllocationShape, Qwen3CpuRequirements};

const ARCHITECTURE: &str = "general.architecture";
const BLOCK_COUNT: &str = "qwen3.block_count";
const CONTEXT_LENGTH: &str = "qwen3.context_length";
const HIDDEN: &str = "qwen3.embedding_length";
const FEED_FORWARD: &str = "qwen3.feed_forward_length";
const HEADS: &str = "qwen3.attention.head_count";
const KV_HEADS: &str = "qwen3.attention.head_count_kv";
const KEY_LENGTH: &str = "qwen3.attention.key_length";
const VALUE_LENGTH: &str = "qwen3.attention.value_length";
const RMS_EPSILON: &str = "qwen3.attention.layer_norm_rms_epsilon";
const CAUSAL: &str = "qwen3.attention.causal";
const ROPE_DIMENSION: &str = "qwen3.rope.dimension_count";
const ROPE_BASE: &str = "qwen3.rope.freq_base";
const ROPE_SCALING_TYPE: &str = "qwen3.rope.scaling.type";
const ROPE_SCALING_FACTOR: &str = "qwen3.rope.scaling.factor";
const ROPE_SCALING_ATTENTION_FACTOR: &str = "qwen3.rope.scaling.attn_factor";
const ROPE_LEGACY_LINEAR_SCALE: &str = "qwen3.rope.scale_linear";
const POOLING_TYPE: &str = "qwen3.pooling_type";

const TOKEN_EMBEDDING: &str = "token_embd.weight";
const OUTPUT_NORM: &str = "output_norm.weight";
const LAST_POOLING_TYPE: u64 = 3;
pub(crate) const RANK_HEAD: &str = "cls.output.weight";
pub(crate) const RANK_LABELS_KEY: &str = "qwen3.classifier.output_labels";
pub(crate) const RANK_LABELS: [&str; 2] = ["yes", "no"];

#[derive(Clone, Copy)]
pub(crate) enum Qwen3Profile {
    Embedding,
    Rank,
}

impl Qwen3Profile {
    const fn pooling_type(self) -> u64 {
        match self {
            Self::Embedding => LAST_POOLING_TYPE,
            Self::Rank => 4,
        }
    }

    const fn profile_roles(self) -> &'static [Role] {
        match self {
            Self::Embedding => &[],
            Self::Rank => &[RANK_HEAD_ROLE],
        }
    }
}

#[derive(Clone, Copy)]
enum Shape {
    HiddenVocabulary,
    Hidden,
    Head,
    HiddenQuery,
    HiddenKeyValue,
    QueryHidden,
    HiddenFeedForward,
    FeedForwardHidden,
    HiddenRankLabels,
}

#[derive(Clone, Copy)]
enum Storage {
    F32Vector,
    F32OrQ8Matrix,
}

#[derive(Clone, Copy)]
pub(crate) struct Role {
    name: &'static str,
    shape: Shape,
    storage: Storage,
}

pub(crate) const RANK_HEAD_ROLE: Role = Role {
    name: RANK_HEAD,
    shape: Shape::HiddenRankLabels,
    storage: Storage::F32OrQ8Matrix,
};

const GLOBAL_ROLES: &[Role] = &[
    Role {
        name: TOKEN_EMBEDDING,
        shape: Shape::HiddenVocabulary,
        storage: Storage::F32OrQ8Matrix,
    },
    Role {
        name: OUTPUT_NORM,
        shape: Shape::Hidden,
        storage: Storage::F32Vector,
    },
];

const BLOCK_ROLES: &[Role] = &[
    Role {
        name: "attn_norm.weight",
        shape: Shape::Hidden,
        storage: Storage::F32Vector,
    },
    Role {
        name: "attn_q_norm.weight",
        shape: Shape::Head,
        storage: Storage::F32Vector,
    },
    Role {
        name: "attn_k_norm.weight",
        shape: Shape::Head,
        storage: Storage::F32Vector,
    },
    Role {
        name: "ffn_norm.weight",
        shape: Shape::Hidden,
        storage: Storage::F32Vector,
    },
    Role {
        name: "attn_q.weight",
        shape: Shape::HiddenQuery,
        storage: Storage::F32OrQ8Matrix,
    },
    Role {
        name: "attn_k.weight",
        shape: Shape::HiddenKeyValue,
        storage: Storage::F32OrQ8Matrix,
    },
    Role {
        name: "attn_v.weight",
        shape: Shape::HiddenKeyValue,
        storage: Storage::F32OrQ8Matrix,
    },
    Role {
        name: "attn_output.weight",
        shape: Shape::QueryHidden,
        storage: Storage::F32OrQ8Matrix,
    },
    Role {
        name: "ffn_gate.weight",
        shape: Shape::HiddenFeedForward,
        storage: Storage::F32OrQ8Matrix,
    },
    Role {
        name: "ffn_up.weight",
        shape: Shape::HiddenFeedForward,
        storage: Storage::F32OrQ8Matrix,
    },
    Role {
        name: "ffn_down.weight",
        shape: Shape::FeedForwardHidden,
        storage: Storage::F32OrQ8Matrix,
    },
];

/// Shared, verified Qwen3 causal body with its checked geometry.
#[derive(Debug)]
pub(crate) struct Qwen3BodyWeights<'artifact> {
    payload: &'artifact VerifiedArtifact,
    layout: Layout,
}

impl<'artifact> Qwen3BodyWeights<'artifact> {
    pub(crate) fn try_from_verified(
        payload: &'artifact VerifiedArtifact,
        profile: Qwen3Profile,
    ) -> Result<Self> {
        let layout = Layout::from_artifact(payload, profile)?;
        validate_inventory(
            payload.observation().tensor_descriptors(),
            layout,
            profile.profile_roles(),
        )?;
        Ok(Self { payload, layout })
    }

    pub(crate) const fn hidden_width(&self) -> usize {
        self.layout.hidden
    }

    pub(crate) const fn max_context(&self) -> usize {
        self.layout.context
    }

    pub(crate) const fn payload(&self) -> &'artifact VerifiedArtifact {
        self.payload
    }

    pub(crate) fn execution(&self, max_context: usize) -> Result<Qwen3Execution<'_, 'artifact>> {
        if max_context == 0 || max_context > self.max_context() {
            return Qwen3ExecutionSnafu {
                requested: max_context,
                rule: "max context must be nonzero and no greater than the artifact context",
            }
            .fail();
        }
        Ok(Qwen3Execution {
            body: self,
            max_context,
        })
    }
}

/// One verified Qwen3 embedding payload with its checked causal geometry.
#[derive(Debug)]
pub struct Qwen3Weights<'artifact> {
    body: Qwen3BodyWeights<'artifact>,
}

impl<'artifact> Qwen3Weights<'artifact> {
    /// Bind one digest-verified GGUF payload to the bounded Qwen3 embedding profile.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when metadata, tensor roles, shapes, or the
    /// bounded no-output-head profile are not satisfied.
    pub fn try_from_verified(payload: &'artifact VerifiedArtifact) -> Result<Self> {
        Ok(Self {
            body: Qwen3BodyWeights::try_from_verified(payload, Qwen3Profile::Embedding)?,
        })
    }

    /// Return the artifact-derived hidden-vector width.
    #[must_use]
    pub const fn hidden_width(&self) -> usize {
        self.body.hidden_width()
    }

    /// Return the artifact-derived maximum context length.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.body.max_context()
    }

    /// Create one stateless bounded CPU embedding executor.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `max_context` is zero or exceeds the
    /// verified artifact's declared context length.
    pub fn execution(&self, max_context: usize) -> Result<Qwen3Execution<'_, 'artifact>> {
        self.body.execution(max_context)
    }
}

/// Stateless Qwen3 causal execution with an explicit caller context bound.
#[derive(Debug)]
pub struct Qwen3Execution<'weights, 'artifact> {
    body: &'weights Qwen3BodyWeights<'artifact>,
    max_context: usize,
}

impl Qwen3Execution<'_, '_> {
    /// Execute token IDs through every causal block and return the final RMS-normalized last row.
    ///
    /// The input contains no padding: every supplied ID is a valid token row
    /// and the final supplied ID selects the returned hidden vector. Pooling,
    /// prompts, tokenizer policy, and output L2 normalization remain outside
    /// this family execution boundary.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] without exposing partial hidden states when an
    /// input, artifact row, allocation, or finite arithmetic check fails.
    pub fn last_hidden(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() || token_ids.len() > self.max_context {
            return Qwen3ExecutionSnafu {
                requested: token_ids.len(),
                rule: "token IDs must be nonempty and fit the caller context bound",
            }
            .fail();
        }
        let shape = self.allocation_shape(token_ids.len())?;
        let embedding = CheckedMatrix::from_payload(self.body.payload(), TOKEN_EMBEDDING)?;
        let mut hidden = reserve("token hidden rows", shape.hidden_rows)?;
        for token_id in token_ids {
            let token = usize::try_from(*token_id).map_err(|_| {
                Qwen3ExecutionSnafu {
                    requested: usize::MAX,
                    rule: "token ID must fit usize",
                }
                .build()
            })?;
            let embedding_row = embedding.decode_row(token)?;
            if embedding_row.len() != shape.decoded_embedding_row {
                return Qwen3ExecutionSnafu {
                    requested: embedding_row.len(),
                    rule: "decoded embedding row must fit the checked Qwen3 allocation shape",
                }
                .fail();
            }
            hidden.extend(embedding_row);
        }
        for block in 0..self.body.layout.blocks {
            self.run_block(block, &mut hidden, &shape)?;
        }
        let final_norm = read_f32_vector(self.body.payload(), OUTPUT_NORM, shape.final_norm)?;
        let normalized = kernels::cpu_f32::rms_norm(
            &hidden,
            &final_norm,
            shape.tokens,
            shape.hidden,
            self.body.layout.epsilon,
        )
        .context(Qwen3CpuSnafu)?;
        let start = product(shape.tokens - 1, shape.hidden)?;
        let result = normalized.get(start..).ok_or_else(|| {
            Qwen3ExecutionSnafu {
                requested: start,
                rule: "final hidden row must fit normalized token rows",
            }
            .build()
        })?;
        finite(result, "final RMS norm")?;
        if result.len() != shape.returned_hidden {
            return Qwen3ExecutionSnafu {
                requested: result.len(),
                rule: "final hidden row must fit the checked Qwen3 allocation shape",
            }
            .fail();
        }
        let mut output = reserve("final hidden row", shape.returned_hidden)?;
        output.extend_from_slice(result);
        Ok(output)
    }

    /// Return the checked CPU allocation envelope for this executor's admitted context.
    ///
    /// The envelope is a logical `f32` backing bound, not allocator capacity,
    /// process RSS, tokenizer storage, GPU memory, or decoded GGUF weight storage.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the admitted allocation dimensions or
    /// their logical byte totals cannot be represented safely.
    pub fn cpu_requirements(&self) -> Result<Qwen3CpuRequirements> {
        let shape = self.allocation_shape(self.max_context)?;
        let inspection = self.body.payload().observation().inspection();
        Qwen3CpuRequirements::embedding(
            inspection.digest,
            inspection.file_len,
            self.max_context,
            &shape,
        )
    }

    pub(crate) fn allocation_shape(&self, tokens: usize) -> Result<Qwen3AllocationShape> {
        self.body.layout.allocation_shape(tokens)
    }

    pub(crate) const fn admitted_max_context(&self) -> usize {
        self.max_context
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the checked causal-attention and FFN order is one source-defined transformer block"
    )]
    fn run_block(
        &self,
        block: usize,
        hidden: &mut [f32],
        shape: &Qwen3AllocationShape,
    ) -> Result<()> {
        let layout = self.body.layout;
        if hidden.len() != shape.hidden_rows {
            return Qwen3ExecutionSnafu {
                requested: hidden.len(),
                rule: "block hidden rows must fit the checked Qwen3 allocation shape",
            }
            .fail();
        }
        let tokens = shape.tokens;
        let attn_norm = read_f32_vector(
            self.body.payload(),
            &block_name(block, "attn_norm.weight"),
            shape.attention_norm,
        )?;
        let q_norm = read_f32_vector(
            self.body.payload(),
            &block_name(block, "attn_q_norm.weight"),
            shape.query_norm,
        )?;
        let k_norm = read_f32_vector(
            self.body.payload(),
            &block_name(block, "attn_k_norm.weight"),
            shape.key_norm,
        )?;
        let q =
            CheckedMatrix::from_payload(self.body.payload(), &block_name(block, "attn_q.weight"))?;
        let k =
            CheckedMatrix::from_payload(self.body.payload(), &block_name(block, "attn_k.weight"))?;
        let v =
            CheckedMatrix::from_payload(self.body.payload(), &block_name(block, "attn_v.weight"))?;
        let output = CheckedMatrix::from_payload(
            self.body.payload(),
            &block_name(block, "attn_output.weight"),
        )?;
        let mut keys = reserve("causal key cache", shape.key_cache)?;
        let mut values = reserve("causal value cache", shape.value_cache)?;
        keys.resize(shape.key_cache, 0.0);
        values.resize(shape.value_cache, 0.0);
        let mut attention = reserve("attention residual", shape.attention_residual)?;
        for token in 0..tokens {
            let row = row(hidden, token, layout.hidden)?;
            let normalized = kernels::cpu_f32::rms_norm(
                row,
                &attn_norm,
                1,
                shape.attention_row_norm,
                layout.epsilon,
            )
            .context(Qwen3CpuSnafu)?;
            let mut query = q.project(&normalized)?;
            let mut key = k.project(&normalized)?;
            let value = v.project(&normalized)?;
            query = kernels::cpu_f32::rms_norm(
                &query,
                &q_norm,
                shape.heads,
                shape.head_dim,
                layout.epsilon,
            )
            .context(Qwen3CpuSnafu)?;
            key = kernels::cpu_f32::rms_norm(
                &key,
                &k_norm,
                shape.kv_heads,
                shape.head_dim,
                layout.epsilon,
            )
            .context(Qwen3CpuSnafu)?;
            apply_neox_rope(&mut query, token, layout)?;
            apply_neox_rope(&mut key, token, layout)?;
            let cache_start = product(token, layout.kv_width)?;
            copy_into(&mut keys, cache_start, &key, "key cache")?;
            copy_into(&mut values, cache_start, &value, "value cache")?;
            let merged = causal_attention(&query, &keys, &values, token + 1, layout, shape)?;
            attention.extend(output.project(&merged)?);
        }
        add_in_place(hidden, &attention, "attention residual")?;
        let ffn_norm = read_f32_vector(
            self.body.payload(),
            &block_name(block, "ffn_norm.weight"),
            shape.ffn_norm,
        )?;
        let gate = CheckedMatrix::from_payload(
            self.body.payload(),
            &block_name(block, "ffn_gate.weight"),
        )?;
        let up =
            CheckedMatrix::from_payload(self.body.payload(), &block_name(block, "ffn_up.weight"))?;
        let down = CheckedMatrix::from_payload(
            self.body.payload(),
            &block_name(block, "ffn_down.weight"),
        )?;
        let mut ffn = reserve("FFN residual", shape.ffn_residual)?;
        for token in 0..tokens {
            let normalized = kernels::cpu_f32::rms_norm(
                row(hidden, token, layout.hidden)?,
                &ffn_norm,
                1,
                shape.ffn_row_norm,
                layout.epsilon,
            )
            .context(Qwen3CpuSnafu)?;
            let activated =
                kernels::cpu_f32::try_silu(&gate.project(&normalized)?).context(Qwen3CpuSnafu)?;
            let up = up.project(&normalized)?;
            let fused = kernels::cpu_f32::try_hadamard(&activated, &up).context(Qwen3CpuSnafu)?;
            finite(&fused, "SwiGLU")?;
            ffn.extend(down.project(&fused)?);
        }
        add_in_place(hidden, &ffn, "FFN residual")
    }
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    blocks: usize,
    context: usize,
    hidden: usize,
    feed_forward: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_width: usize,
    q_width: usize,
    vocabulary: usize,
    epsilon: f32,
    rope_base: f64,
}

impl Layout {
    #[expect(
        clippy::too_many_lines,
        reason = "strict metadata parsing keeps Qwen3 profile relations in one auditable authority"
    )]
    fn from_artifact(payload: &VerifiedArtifact, profile: Qwen3Profile) -> Result<Self> {
        let metadata = payload.observation().metadata();
        require_string(metadata, ARCHITECTURE, "qwen3")?;
        require_u32(metadata, POOLING_TYPE, "profile pooling type")?
            .eq(&profile.pooling_type())
            .then_some(())
            .ok_or_else(|| {
                Qwen3MetadataSnafu {
                    key: POOLING_TYPE,
                    rule: "must select the requested bounded profile pooling type",
                }
                .build()
            })?;
        if matches!(metadata.get(CAUSAL), Some(MetaValue::Bool(false))) {
            return Qwen3MetadataSnafu {
                key: CAUSAL,
                rule: "explicit false is outside the bounded causal profile",
            }
            .fail();
        }
        if metadata
            .get(CAUSAL)
            .is_some_and(|value| !matches!(value, MetaValue::Bool(_)))
        {
            return Qwen3MetadataSnafu {
                key: CAUSAL,
                rule: "must be boolean when present",
            }
            .fail();
        }
        require_neutral_rope_scaling(metadata)?;
        let blocks = positive(metadata, BLOCK_COUNT)?;
        let context = positive(metadata, CONTEXT_LENGTH)?;
        let hidden = positive(metadata, HIDDEN)?;
        let feed_forward = positive(metadata, FEED_FORWARD)?;
        let heads = positive(metadata, HEADS)?;
        let kv_heads = positive(metadata, KV_HEADS)?;
        let head_dim = positive(metadata, KEY_LENGTH)?;
        let value_dim = positive(metadata, VALUE_LENGTH)?;
        let rope_dim = positive(metadata, ROPE_DIMENSION)?;
        let epsilon = require_f32(metadata, RMS_EPSILON)?;
        let rope_base = f64::from(require_f32(metadata, ROPE_BASE)?);
        if !heads.is_multiple_of(kv_heads) {
            return Qwen3MetadataSnafu {
                key: HEADS,
                rule: "must divide evenly by Q/KV head count",
            }
            .fail();
        }
        if value_dim != head_dim {
            return Qwen3MetadataSnafu {
                key: VALUE_LENGTH,
                rule: "must equal the key/head width in this bounded path",
            }
            .fail();
        }
        if rope_dim != head_dim || !rope_dim.is_multiple_of(2) {
            return Qwen3MetadataSnafu {
                key: ROPE_DIMENSION,
                rule: "must equal the full even key/head width in this bounded path",
            }
            .fail();
        }
        if !epsilon.is_finite() || epsilon <= 0.0 || !rope_base.is_finite() || rope_base <= 0.0 {
            return Qwen3MetadataSnafu {
                key: RMS_EPSILON,
                rule: "epsilon and RoPE base must be finite positive values",
            }
            .fail();
        }
        let q_width = product(heads, head_dim)?;
        let kv_width = product(kv_heads, head_dim)?;
        let vocabulary = vocabulary(metadata)?;
        Ok(Self {
            blocks,
            context,
            hidden,
            feed_forward,
            heads,
            kv_heads,
            head_dim,
            kv_width,
            q_width,
            vocabulary,
            epsilon,
            rope_base,
        })
    }

    fn allocation_shape(self, tokens: usize) -> Result<Qwen3AllocationShape> {
        Qwen3AllocationShape::new(
            tokens,
            self.hidden,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.q_width,
            self.kv_width,
            self.feed_forward,
            RANK_LABELS.len(),
        )
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one role inventory defines the complete bounded Qwen3 embedding tensor contract"
)]
fn validate_inventory(
    tensors: &[loader::gguf::TensorDescriptor],
    layout: Layout,
    profile_roles: &[Role],
) -> Result<()> {
    let expected_count = layout
        .blocks
        .checked_mul(BLOCK_ROLES.len())
        .and_then(|count| count.checked_add(GLOBAL_ROLES.len()))
        .and_then(|count| count.checked_add(profile_roles.len()))
        .ok_or_else(|| {
            Qwen3ExecutionSnafu {
                requested: layout.blocks,
                rule: "tensor inventory count overflowed",
            }
            .build()
        })?;
    if tensors.len() != expected_count {
        return Qwen3TensorSnafu {
            name: "inventory".to_string(),
            rule: "must have exactly the metadata-derived bounded role count",
        }
        .fail();
    }
    let mut found = HashSet::new();
    found
        .try_reserve(expected_count)
        .context(Qwen3AllocationSnafu {
            target: "Qwen3 tensor inventory",
            length: expected_count,
        })?;
    for tensor in tensors {
        let Some(role) = role_for_name(&tensor.name, layout.blocks, profile_roles) else {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "is outside the bounded profile role inventory",
            }
            .fail();
        };
        if !found.insert(tensor.name.clone()) {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "must not occur more than once",
            }
            .fail();
        }
        if tensor.dims != role.shape.dimensions(layout)? {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "shape must derive from checked architecture metadata",
            }
            .fail();
        }
        if matches!(role.storage, Storage::F32Vector) && tensor.ggml_type != GgmlType::F32 {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "normalization vectors must use F32",
            }
            .fail();
        }
        if matches!(role.storage, Storage::F32OrQ8Matrix)
            && !matches!(tensor.ggml_type, GgmlType::F32 | GgmlType::Q8_0)
        {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "matrices must use the bounded F32 or Q8_0 executable storage",
            }
            .fail();
        }
    }
    for role in GLOBAL_ROLES {
        if !found.contains(role.name) {
            return Qwen3TensorSnafu {
                name: role.name.to_string(),
                rule: "is required by the bounded profile",
            }
            .fail();
        }
    }
    for role in profile_roles {
        if !found.contains(role.name) {
            return Qwen3TensorSnafu {
                name: role.name.to_string(),
                rule: "is required by the bounded Qwen3 profile",
            }
            .fail();
        }
    }
    for block in 0..layout.blocks {
        for role in BLOCK_ROLES {
            let name = block_name(block, role.name);
            if !found.contains(&name) {
                return Qwen3TensorSnafu {
                    name,
                    rule: "is required by the bounded profile",
                }
                .fail();
            }
        }
    }
    Ok(())
}

impl Shape {
    fn dimensions(self, layout: Layout) -> Result<Vec<u64>> {
        let hidden = u64_from(layout.hidden)?;
        let vocabulary = u64_from(layout.vocabulary)?;
        let head = u64_from(layout.head_dim)?;
        let query = u64_from(layout.q_width)?;
        let key_value = u64_from(layout.kv_width)?;
        let feed_forward = u64_from(layout.feed_forward)?;
        Ok(match self {
            Self::HiddenVocabulary => vec![hidden, vocabulary],
            Self::Hidden => vec![hidden],
            Self::Head => vec![head],
            Self::HiddenQuery => vec![hidden, query],
            Self::HiddenKeyValue => vec![hidden, key_value],
            Self::QueryHidden => vec![query, hidden],
            Self::HiddenFeedForward => vec![hidden, feed_forward],
            Self::FeedForwardHidden => vec![feed_forward, hidden],
            Self::HiddenRankLabels => vec![hidden, u64_from(RANK_LABELS.len())?],
        })
    }
}

fn role_for_name(name: &str, blocks: usize, profile_roles: &[Role]) -> Option<Role> {
    if let Some(role) = GLOBAL_ROLES.iter().copied().find(|role| role.name == name) {
        return Some(role);
    }
    if let Some(role) = profile_roles.iter().copied().find(|role| role.name == name) {
        return Some(role);
    }
    let remainder = name.strip_prefix("blk.")?;
    let (block, role_name) = remainder.split_once('.')?;
    let block = block.parse::<usize>().ok()?;
    if block >= blocks {
        return None;
    }
    BLOCK_ROLES
        .iter()
        .copied()
        .find(|role| role.name == role_name)
}

fn causal_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    tokens: usize,
    layout: Layout,
    shape: &Qwen3AllocationShape,
) -> Result<Vec<f32>> {
    let prefix = shape.causal_prefix_elements(tokens)?;
    let mut output = reserve("causal attention output", shape.causal_attention_output)?;
    let group = layout.heads / layout.kv_heads;
    let head_dim = layout.head_dim.to_f32().ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: layout.head_dim,
            rule: "head dimension must convert to f32 for attention scaling",
        }
        .build()
    })?;
    let scale = head_dim.sqrt().recip();
    for head in 0..layout.heads {
        let q = row(query, head, layout.head_dim)?;
        let kv_head = head / group;
        let mut scores = reserve("causal attention scores", prefix)?;
        for token in 0..prefix {
            let key = row(row(keys, token, layout.kv_width)?, kv_head, layout.head_dim)?;
            let score = q
                .iter()
                .zip(key)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale;
            finite_one(score, "attention score", token)?;
            scores.push(score);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exponents = reserve("causal attention exponentials", prefix)?;
        for score in &scores {
            let exponent = (*score - max).exp();
            finite_one(exponent, "attention exponential", exponents.len())?;
            exponents.push(exponent);
        }
        let total = exponents.iter().sum::<f32>();
        finite_one(total, "attention normalization", head)?;
        for lane in 0..layout.head_dim {
            let mut value = 0.0;
            for (token, exponent) in exponents.iter().enumerate() {
                let v = row(
                    row(values, token, layout.kv_width)?,
                    kv_head,
                    layout.head_dim,
                )?[lane];
                value += exponent / total * v;
            }
            finite_one(value, "attention value", head * layout.head_dim + lane)?;
            output.push(value);
        }
    }
    Ok(output)
}

fn apply_neox_rope(values: &mut [f32], position: usize, layout: Layout) -> Result<()> {
    let position = position.to_f64().ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: position,
            rule: "RoPE position must convert to f64",
        }
        .build()
    })?;
    let head_dim = layout.head_dim.to_f64().ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: layout.head_dim,
            rule: "RoPE head dimension must convert to f64",
        }
        .build()
    })?;
    for head in values.chunks_exact_mut(layout.head_dim) {
        for pair_index in 0..layout.head_dim / 2 {
            let pair = pair_index.to_f64().ok_or_else(|| {
                Qwen3ExecutionSnafu {
                    requested: layout.head_dim,
                    rule: "RoPE pair index must convert to f64",
                }
                .build()
            })?;
            let angle = position / layout.rope_base.powf((2.0 * pair) / head_dim);
            let cosine = angle.cos().to_f32().ok_or_else(|| {
                Qwen3ArithmeticSnafu {
                    stage: "NeoX RoPE cosine",
                    index: pair_index,
                }
                .build()
            })?;
            let sine = angle.sin().to_f32().ok_or_else(|| {
                Qwen3ArithmeticSnafu {
                    stage: "NeoX RoPE sine",
                    index: pair_index,
                }
                .build()
            })?;
            let right = pair_index + layout.head_dim / 2;
            let (left_value, right_value) = (head[pair_index], head[right]);
            head[pair_index] = left_value * cosine - right_value * sine;
            head[right] = left_value * sine + right_value * cosine;
        }
    }
    finite(values, "NeoX RoPE")
}

fn read_f32_vector(payload: &VerifiedArtifact, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = payload.tensor(name).map_err(|_| {
        Qwen3TensorSnafu {
            name: name.to_string(),
            rule: "must be present in the verified payload",
        }
        .build()
    })?;
    if tensor.ggml_type() != GgmlType::F32 || tensor.dims() != [u64_from(width)?] {
        return Qwen3TensorSnafu {
            name: name.to_string(),
            rule: "must be an F32 vector with its metadata-derived width",
        }
        .fail();
    }
    quant::row_decode_f32(quant::RowFormat::F32, tensor.bytes(), width).map_err(|_| {
        Qwen3TensorSnafu {
            name: name.to_string(),
            rule: "must contain one finite complete F32 row",
        }
        .build()
    })
}

fn require_string(
    metadata: &HashMap<String, MetaValue>,
    key: &'static str,
    expected: &'static str,
) -> Result<()> {
    match metadata.get(key) {
        Some(MetaValue::String(value)) if value == expected => Ok(()),
        _ => Qwen3MetadataSnafu {
            key,
            rule: "must be the exact bounded profile value",
        }
        .fail(),
    }
}
fn require_u32(
    metadata: &HashMap<String, MetaValue>,
    key: &'static str,
    rule: &'static str,
) -> Result<u64> {
    match metadata.get(key) {
        Some(MetaValue::U32(value)) => Ok(u64::from(*value)),
        _ => Qwen3MetadataSnafu { key, rule }.fail(),
    }
}
fn require_f32(metadata: &HashMap<String, MetaValue>, key: &'static str) -> Result<f32> {
    match metadata.get(key) {
        Some(MetaValue::F32(value)) => Ok(*value),
        _ => Qwen3MetadataSnafu {
            key,
            rule: "must be F32",
        }
        .fail(),
    }
}
fn positive(metadata: &HashMap<String, MetaValue>, key: &'static str) -> Result<usize> {
    let value = require_u32(metadata, key, "must be U32")?;
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            Qwen3MetadataSnafu {
                key,
                rule: "must be a positive usize",
            }
            .build()
        })
}
fn vocabulary(metadata: &HashMap<String, MetaValue>) -> Result<usize> {
    match metadata.get("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(values)) if values.element_type() == MetaValueType::String => {
            NonZeroUsize::new(values.values().len())
                .map(NonZeroUsize::get)
                .ok_or_else(|| {
                    Qwen3MetadataSnafu {
                        key: "tokenizer.ggml.tokens",
                        rule: "must be a nonempty string array",
                    }
                    .build()
                })
        }
        _ => Qwen3MetadataSnafu {
            key: "tokenizer.ggml.tokens",
            rule: "must be a string array",
        }
        .fail(),
    }
}
fn require_neutral_rope_scaling(metadata: &HashMap<String, MetaValue>) -> Result<()> {
    let scaling_none = match metadata.get(ROPE_SCALING_TYPE) {
        None => false,
        Some(MetaValue::String(value)) if value == "linear" => false,
        Some(MetaValue::String(value)) if value == "none" => true,
        _ => {
            return Qwen3MetadataSnafu {
                key: ROPE_SCALING_TYPE,
                rule: "must be absent, linear, or source-neutral none scaling",
            }
            .fail();
        }
    };
    let current = optional_finite_f32(metadata, ROPE_SCALING_FACTOR)?;
    let legacy = optional_finite_f32(metadata, ROPE_LEGACY_LINEAR_SCALE)?;
    if !scaling_none {
        let (key, factor) = match current {
            Some(factor) => (ROPE_SCALING_FACTOR, factor),
            None => (ROPE_LEGACY_LINEAR_SCALE, legacy.unwrap_or(0.0)),
        };
        if factor.to_bits() != 0.0_f32.to_bits() && factor.to_bits() != 1.0_f32.to_bits() {
            return Qwen3MetadataSnafu {
                key,
                rule: "must resolve to source-neutral linear factor 0 or 1",
            }
            .fail();
        }
    }
    if let Some(attention) = optional_finite_f32(metadata, ROPE_SCALING_ATTENTION_FACTOR)?
        && attention.to_bits() != 1.0_f32.to_bits()
    {
        return Qwen3MetadataSnafu {
            key: ROPE_SCALING_ATTENTION_FACTOR,
            rule: "must be absent or exact neutral attention factor 1",
        }
        .fail();
    }
    Ok(())
}
fn optional_finite_f32(
    metadata: &HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Option<f32>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetaValue::F32(value)) if value.is_finite() => Ok(Some(*value)),
        _ => Qwen3MetadataSnafu {
            key,
            rule: "must be finite F32 when present",
        }
        .fail(),
    }
}
fn block_name(block: usize, role: &str) -> String {
    format!("blk.{block}.{role}")
}
fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: left,
            rule: "execution geometry multiplication overflowed",
        }
        .build()
    })
}
fn u64_from(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        Qwen3ExecutionSnafu {
            requested: value,
            rule: "execution geometry exceeds u64",
        }
        .build()
    })
}
fn reserve(target: &'static str, length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .context(Qwen3AllocationSnafu { target, length })?;
    Ok(values)
}
fn row(values: &[f32], index: usize, width: usize) -> Result<&[f32]> {
    let start = product(index, width)?;
    values.get(start..start + width).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: index,
            rule: "row must fit its checked buffer",
        }
        .build()
    })
}
fn copy_into(
    destination: &mut [f32],
    start: usize,
    source: &[f32],
    target: &'static str,
) -> Result<()> {
    let end = start.checked_add(source.len()).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: start,
            rule: "copy range overflowed",
        }
        .build()
    })?;
    let slot = destination.get_mut(start..end).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: start,
            rule: "copy range must fit its checked buffer",
        }
        .build()
    })?;
    slot.copy_from_slice(source);
    finite(slot, target)
}
fn add_in_place(destination: &mut [f32], source: &[f32], stage: &'static str) -> Result<()> {
    if destination.len() != source.len() {
        return Qwen3ExecutionSnafu {
            requested: destination.len(),
            rule: "residual operands must have equal lengths",
        }
        .fail();
    }
    for (index, (left, right)) in destination.iter_mut().zip(source).enumerate() {
        *left += right;
        finite_one(*left, stage, index)?;
    }
    Ok(())
}
fn finite(values: &[f32], stage: &'static str) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        finite_one(*value, stage, index)?;
    }
    Ok(())
}
fn finite_one(value: f32, stage: &'static str, index: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Qwen3ArithmeticSnafu { stage, index }.fail()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use loader::gguf::{ArtifactByteLimit, Sha256Digest, VerifiedArtifact};
    use test_fixtures::{RawGguf, RawMetadata, RawMetadataValue, RawTensor, serialize_raw_gguf};

    use super::*;

    const TEST_HIDDEN: u64 = 3;
    const TEST_HEADS: u64 = 2;
    const TEST_KV_HEADS: u64 = 1;
    const TEST_HEAD_DIM: u64 = 2;
    const TEST_FEED_FORWARD: u64 = 4;
    const TEST_VOCABULARY: u64 = 4;
    const TEST_BLOCKS: u64 = 1;
    const TEST_CONTEXT: u32 = 4;

    #[test]
    fn executes_an_asymmetric_causal_fixture_to_a_final_hidden_row()
    -> std::result::Result<(), String> {
        let raw = fixture()?;
        let artifact = verify(&raw)?;
        let weights =
            Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let execution = weights
            .execution(usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let one = execution
            .last_hidden(&[0])
            .map_err(|error| error.to_string())?;
        let two = execution
            .last_hidden(&[0, 1])
            .map_err(|error| error.to_string())?;
        if one.len() != usize::try_from(TEST_HIDDEN).map_err(|error| error.to_string())? {
            return Err("one-token result did not retain the hidden width".to_string());
        }
        if !two.iter().all(|value| value.is_finite()) {
            return Err("final hidden row was not finite".to_string());
        }
        if one == two {
            return Err("causal fixture did not distinguish its final token row".to_string());
        }
        Ok(())
    }

    #[test]
    fn reports_the_executor_admitted_context_and_returned_hidden_vector()
    -> std::result::Result<(), String> {
        let raw = fixture()?;
        let artifact = verify(&raw)?;
        let weights =
            Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let execution = weights.execution(2).map_err(|error| error.to_string())?;
        let inspection = artifact.observation().inspection();
        let requirements = execution
            .cpu_requirements()
            .map_err(|error| error.to_string())?;
        let repeated = execution
            .cpu_requirements()
            .map_err(|error| error.to_string())?;
        if requirements != repeated {
            return Err("requirements were not stable across repeated inspection".to_string());
        }
        if requirements.max_context() != 2 {
            return Err("requirements did not retain the executor context bound".to_string());
        }
        if requirements.artifact_digest() != inspection.digest
            || requirements.serialized_backing_bytes() != inspection.file_len
        {
            return Err("requirements did not bind the verified artifact inspection".to_string());
        }
        let hidden = execution
            .last_hidden(&[0, 1])
            .map_err(|error| error.to_string())?;
        let returned_bytes = u64::try_from(std::mem::size_of_val(hidden.as_slice()))
            .map_err(|error| error.to_string())?;
        if requirements.returned_output_bytes() != returned_bytes {
            return Err(
                "requirements did not retain the actual final hidden Vec backing".to_string(),
            );
        }
        if requirements.workspace_upper_bound_bytes() == 0 {
            return Err("requirements omitted transient workspace".to_string());
        }
        if requirements.logical_f32_upper_bound_bytes()
            != requirements.workspace_upper_bound_bytes() + requirements.returned_output_bytes()
        {
            return Err(
                "embedding logical f32 total did not include the returned vector".to_string(),
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_an_output_head_outside_the_embedding_profile() -> std::result::Result<(), String> {
        let mut raw = fixture()?;
        raw.tensors.push(tensor(
            "output.weight",
            vec![TEST_HIDDEN, TEST_VOCABULARY],
            0.5,
        )?);
        let serialized = serialize_raw_gguf(&raw).map_err(|error| error.to_string())?;
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        let path = directory.path().join("qwen3-extra-output.gguf");
        std::fs::write(&path, serialized.bytes).map_err(|error| error.to_string())?;
        let limit = NonZeroU64::new(serialized.byte_len + 1).ok_or("invalid fixture limit")?;
        let artifact = VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(serialized.sha256),
            ArtifactByteLimit::new(limit),
        )
        .map_err(|error| error.to_string())?;
        if Qwen3Weights::try_from_verified(&artifact).is_ok() {
            return Err("bounded embedding profile accepted output.weight".to_string());
        }
        Ok(())
    }

    #[test]
    fn refuses_a_general_pooling_key_without_the_qwen3_contract_key()
    -> std::result::Result<(), String> {
        let mut raw = fixture()?;
        raw.metadata.retain(|entry| entry.key != POOLING_TYPE);
        raw.metadata.push(metadata_u32(
            "general.pooling_type",
            u32::try_from(LAST_POOLING_TYPE).map_err(|error| error.to_string())?,
        ));
        let artifact = verify(&raw)?;
        if Qwen3Weights::try_from_verified(&artifact).is_ok() {
            return Err(
                "general pooling metadata substituted for qwen3 pooling metadata".to_string(),
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_each_strict_metadata_contract_witness() -> std::result::Result<(), String> {
        let cases = [
            ProfileCase::metadata("missing required hidden width", remove_hidden, HIDDEN),
            ProfileCase::metadata("mistyped hidden width", mistype_hidden, HIDDEN),
            ProfileCase::metadata("explicit noncausal", causal_false, CAUSAL),
            ProfileCase::metadata(
                "unknown rope scaling",
                unknown_rope_scaling,
                ROPE_SCALING_TYPE,
            ),
            ProfileCase::metadata(
                "nonneutral legacy rope scaling",
                nonneutral_legacy_rope_scaling,
                ROPE_LEGACY_LINEAR_SCALE,
            ),
            ProfileCase::metadata(
                "nonneutral current rope scaling",
                nonneutral_current_rope_scaling,
                ROPE_SCALING_FACTOR,
            ),
            ProfileCase::metadata(
                "nonneutral RoPE attention scaling",
                nonneutral_rope_attention_scaling,
                ROPE_SCALING_ATTENTION_FACTOR,
            ),
            ProfileCase::metadata("uneven Q/KV heads", uneven_heads, HEADS),
            ProfileCase::metadata("unequal K/V widths", unequal_value_width, VALUE_LENGTH),
            ProfileCase::metadata("partial rotary width", partial_rotary_width, ROPE_DIMENSION),
        ];
        for case in cases {
            assert_profile_case(case)?;
        }
        Ok(())
    }

    #[test]
    fn accepts_source_neutral_rope_scaling_precedence() -> std::result::Result<(), String> {
        for mutate in [
            source_neutral_none_scaling as fn(&mut RawGguf),
            source_neutral_current_factor_precedes_legacy,
            source_neutral_legacy_factor,
        ] {
            let mut raw = fixture()?;
            mutate(&mut raw);
            let artifact = verify(&raw)?;
            Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    #[test]
    fn rejects_each_strict_tensor_contract_witness() -> std::result::Result<(), String> {
        let cases = [
            ProfileCase::tensor("missing required tensor", remove_tensor, "inventory"),
            ProfileCase::tensor("extra output tensor", extra_tensor, "inventory"),
            ProfileCase::tensor(
                "wrong matrix shape",
                wrong_tensor_shape,
                "blk.0.attn_q.weight",
            ),
            ProfileCase::tensor(
                "wrong matrix dtype",
                wrong_matrix_dtype,
                "blk.0.attn_q.weight",
            ),
            ProfileCase::tensor("wrong norm dtype", wrong_norm_dtype, "output_norm.weight"),
        ];
        for case in cases {
            assert_profile_case(case)?;
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_execution_requests_and_token_ids() -> std::result::Result<(), String> {
        let raw = fixture()?;
        let artifact = verify(&raw)?;
        let weights =
            Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        for requested in [
            0,
            usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())? + 1,
        ] {
            if !matches!(
                weights.execution(requested),
                Err(crate::Error::Qwen3Execution { .. })
            ) {
                return Err("invalid maximum context did not return Qwen3Execution".to_string());
            }
        }
        let execution = weights
            .execution(usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        for tokens in [&[][..], &[0, 1, 2, 3, 0][..]] {
            if !matches!(
                execution.last_hidden(tokens),
                Err(crate::Error::Qwen3Execution { .. })
            ) {
                return Err(
                    "empty or over-context token IDs did not return Qwen3Execution".to_string(),
                );
            }
        }
        if !matches!(
            execution
                .last_hidden(&[u32::try_from(TEST_VOCABULARY).map_err(|error| error.to_string())?]),
            Err(crate::Error::ProjectionInputWidth { .. })
        ) {
            return Err(
                "out-of-vocabulary token ID did not preserve checked matrix bounds error"
                    .to_string(),
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_late_nonfinite_weights_without_poisoning_a_pristine_retry()
    -> std::result::Result<(), String> {
        let mut malformed = fixture()?;
        let tensor = find_tensor_mut(&mut malformed, "blk.0.ffn_down.weight")?;
        tensor.payload[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let bad_artifact = verify(&malformed)?;
        let bad_weights =
            Qwen3Weights::try_from_verified(&bad_artifact).map_err(|error| error.to_string())?;
        let bad_execution = bad_weights
            .execution(usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        if !matches!(
            bad_execution.last_hidden(&[0]),
            Err(crate::Error::ProjectionRow { .. })
        ) {
            return Err(
                "late nonfinite FFN weight did not preserve checked-row refusal".to_string(),
            );
        }

        let pristine = fixture()?;
        let artifact = verify(&pristine)?;
        let weights =
            Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let execution = weights
            .execution(usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let result = execution
            .last_hidden(&[0])
            .map_err(|error| error.to_string())?;
        if !result.iter().all(|value| value.is_finite()) {
            return Err("pristine executor retry produced a nonfinite hidden row".to_string());
        }
        Ok(())
    }

    #[test]
    fn executes_the_rank_profile_to_ordered_raw_classifier_logits()
    -> std::result::Result<(), String> {
        let raw = rank_fixture()?;
        let artifact = verify(&raw)?;
        let weights = crate::Qwen3RankWeights::try_from_verified(&artifact)
            .map_err(|error| error.to_string())?;
        if weights.hidden_width()
            != usize::try_from(TEST_HIDDEN).map_err(|error| error.to_string())?
            || weights.max_context()
                != usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?
        {
            return Err("rank profile did not retain its artifact geometry".to_string());
        }
        let execution = weights.execution(2).map_err(|error| error.to_string())?;
        let requirements = execution
            .cpu_requirements()
            .map_err(|error| error.to_string())?;
        let inspection = artifact.observation().inspection();
        if requirements.max_context() != 2
            || requirements.artifact_digest() != inspection.digest
            || requirements.serialized_backing_bytes() != inspection.file_len
        {
            return Err(
                "rank requirements did not bind the admitted verified executor".to_string(),
            );
        }
        let logits = execution
            .last_logits(&[0, 1])
            .map_err(|error| error.to_string())?;
        if requirements.returned_output_bytes() != 0
            || std::mem::size_of_val(&logits) != 2 * std::mem::size_of::<f32>()
        {
            return Err(
                "rank requirements did not preserve stack-array output semantics".to_string(),
            );
        }
        if !logits.iter().all(|value| value.is_finite())
            || logits[0].to_bits() == logits[1].to_bits()
        {
            return Err(
                "rank classifier did not preserve finite distinct yes/no logits".to_string(),
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_rank_profile_metadata_and_inventory_deviations() -> std::result::Result<(), String> {
        let cases = [
            ProfileCase::metadata("embedding pool", rank_pooling_as_embedding, POOLING_TYPE),
            ProfileCase::metadata("missing labels", remove_rank_labels, RANK_LABELS_KEY),
            ProfileCase::metadata("typed labels", mistype_rank_labels, RANK_LABELS_KEY),
            ProfileCase::metadata("reversed labels", reverse_rank_labels, RANK_LABELS_KEY),
            ProfileCase::metadata("wrong labels", wrong_rank_labels, RANK_LABELS_KEY),
            ProfileCase::metadata("extra label", extra_rank_label, RANK_LABELS_KEY),
            ProfileCase::tensor("missing head", remove_rank_head, "inventory"),
            ProfileCase::tensor("wrong head shape", wrong_rank_head_shape, RANK_HEAD),
            ProfileCase::tensor("wrong head dtype", wrong_rank_head_dtype, RANK_HEAD),
            ProfileCase::tensor("extra classifier bias", extra_rank_bias, "inventory"),
            ProfileCase::tensor("extra language head", extra_rank_lm_head, "inventory"),
            ProfileCase::tensor("missing trunk tensor", remove_tensor, "inventory"),
            ProfileCase::metadata("noncausal", rank_causal_false, CAUSAL),
            ProfileCase::metadata(
                "nonneutral rope scaling",
                nonneutral_current_rope_scaling,
                ROPE_SCALING_FACTOR,
            ),
        ];
        for case in cases {
            assert_rank_profile_case(case)?;
        }
        Ok(())
    }

    #[test]
    fn rejects_a_duplicate_rank_head_during_gguf_verification() -> std::result::Result<(), String> {
        let mut raw = rank_fixture()?;
        let Some(head) = raw
            .tensors
            .iter()
            .find(|tensor| tensor.name == RANK_HEAD)
            .cloned()
        else {
            return Err("rank fixture was missing its classifier head".to_string());
        };
        raw.tensors.push(head);
        if verify(&raw).is_ok() {
            return Err("GGUF verification accepted a duplicate rank classifier head".to_string());
        }
        Ok(())
    }

    #[test]
    fn embedding_profile_refuses_an_otherwise_valid_rank_artifact()
    -> std::result::Result<(), String> {
        let artifact = verify(&rank_fixture()?)?;
        if !matches!(
            Qwen3Weights::try_from_verified(&artifact),
            Err(crate::Error::Qwen3Metadata {
                key: POOLING_TYPE,
                ..
            })
        ) {
            return Err("embedding profile accepted a valid rank artifact".to_string());
        }
        Ok(())
    }

    #[test]
    fn rank_head_refuses_nonfinite_and_overflowing_projection_without_poisoning_retry()
    -> std::result::Result<(), String> {
        for value in [f32::NAN, f32::INFINITY, f32::MAX] {
            let mut malformed = rank_fixture()?;
            let head = find_tensor_mut(&mut malformed, RANK_HEAD)?;
            for lane in head.payload[..12].chunks_exact_mut(4) {
                lane.copy_from_slice(&value.to_le_bytes());
            }
            let artifact = verify(&malformed)?;
            let weights = crate::Qwen3RankWeights::try_from_verified(&artifact)
                .map_err(|error| error.to_string())?;
            let execution = weights
                .execution(usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
            match execution.last_logits(&[0]) {
                Err(crate::Error::ProjectionRow { .. } | crate::Error::Qwen3Arithmetic { .. }) => {}
                other => {
                    return Err(format!(
                        "rank head nonfinite or overflowing projection was accepted for {value:?}: {other:?}"
                    ));
                }
            }
        }
        let artifact = verify(&rank_fixture()?)?;
        let logits = crate::Qwen3RankWeights::try_from_verified(&artifact)
            .map_err(|error| error.to_string())?
            .execution(usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?)
            .and_then(|execution| execution.last_logits(&[0]))
            .map_err(|error| error.to_string())?;
        if !logits.iter().all(|value| value.is_finite()) {
            return Err("pristine rank retry produced nonfinite logits".to_string());
        }
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum ExpectedProfileError {
        Metadata(&'static str),
        Tensor(&'static str),
    }

    #[derive(Clone, Copy)]
    struct ProfileCase {
        name: &'static str,
        mutate: fn(&mut RawGguf),
        expected: ExpectedProfileError,
    }

    impl ProfileCase {
        const fn metadata(name: &'static str, mutate: fn(&mut RawGguf), key: &'static str) -> Self {
            Self {
                name,
                mutate,
                expected: ExpectedProfileError::Metadata(key),
            }
        }

        const fn tensor(
            name: &'static str,
            mutate: fn(&mut RawGguf),
            tensor: &'static str,
        ) -> Self {
            Self {
                name,
                mutate,
                expected: ExpectedProfileError::Tensor(tensor),
            }
        }
    }

    fn assert_profile_case(case: ProfileCase) -> std::result::Result<(), String> {
        let mut raw = fixture()?;
        (case.mutate)(&mut raw);
        let error = profile_error(&raw)?;
        let accepted = match case.expected {
            ExpectedProfileError::Metadata(key) => {
                matches!(error, crate::Error::Qwen3Metadata { key: actual, .. } if actual == key)
            }
            ExpectedProfileError::Tensor(name) => {
                matches!(error, crate::Error::Qwen3Tensor { name: ref actual, .. } if actual == name)
            }
        };
        if accepted {
            Ok(())
        } else {
            Err(format!(
                "{} returned the wrong typed profile error: {error}",
                case.name
            ))
        }
    }

    fn assert_rank_profile_case(case: ProfileCase) -> std::result::Result<(), String> {
        let mut raw = rank_fixture()?;
        (case.mutate)(&mut raw);
        let error = rank_profile_error(&raw)?;
        let accepted = match case.expected {
            ExpectedProfileError::Metadata(key) => {
                matches!(error, crate::Error::Qwen3Metadata { key: actual, .. } if actual == key)
            }
            ExpectedProfileError::Tensor(name) => {
                matches!(error, crate::Error::Qwen3Tensor { name: ref actual, .. } if actual == name)
            }
        };
        if accepted {
            Ok(())
        } else {
            Err(format!(
                "rank {} returned the wrong typed profile error: {error}",
                case.name
            ))
        }
    }

    fn profile_error(raw: &RawGguf) -> std::result::Result<crate::Error, String> {
        let artifact = verify(raw)?;
        Qwen3Weights::try_from_verified(&artifact)
            .err()
            .ok_or_else(|| "invalid profile fixture was accepted".to_string())
    }

    fn rank_profile_error(raw: &RawGguf) -> std::result::Result<crate::Error, String> {
        let artifact = verify(raw)?;
        crate::Qwen3RankWeights::try_from_verified(&artifact)
            .err()
            .ok_or_else(|| "invalid rank profile fixture was accepted".to_string())
    }

    fn remove_hidden(raw: &mut RawGguf) {
        raw.metadata.retain(|entry| entry.key != HIDDEN);
    }

    fn rank_pooling_as_embedding(raw: &mut RawGguf) {
        replace_metadata(raw, POOLING_TYPE, &RawMetadataValue::U32(3));
    }

    fn remove_rank_labels(raw: &mut RawGguf) {
        raw.metadata.retain(|entry| entry.key != RANK_LABELS_KEY);
    }

    fn mistype_rank_labels(raw: &mut RawGguf) {
        replace_metadata(
            raw,
            RANK_LABELS_KEY,
            &RawMetadataValue::I32Array(vec![1, 2]),
        );
    }

    fn reverse_rank_labels(raw: &mut RawGguf) {
        replace_metadata(
            raw,
            RANK_LABELS_KEY,
            &RawMetadataValue::StringArray(vec!["no".to_string(), "yes".to_string()]),
        );
    }

    fn wrong_rank_labels(raw: &mut RawGguf) {
        replace_metadata(
            raw,
            RANK_LABELS_KEY,
            &RawMetadataValue::StringArray(vec!["yes".to_string(), "maybe".to_string()]),
        );
    }

    fn wrong_rank_head_shape(raw: &mut RawGguf) {
        replace_tensor_dimensions(raw, RANK_HEAD, &[TEST_HIDDEN, 1]);
    }

    fn extra_rank_label(raw: &mut RawGguf) {
        replace_metadata(
            raw,
            RANK_LABELS_KEY,
            &RawMetadataValue::StringArray(vec![
                "yes".to_string(),
                "no".to_string(),
                "maybe".to_string(),
            ]),
        );
    }

    fn remove_rank_head(raw: &mut RawGguf) {
        raw.tensors.retain(|tensor| tensor.name != RANK_HEAD);
    }

    fn wrong_rank_head_dtype(raw: &mut RawGguf) {
        for tensor in &mut raw.tensors {
            if tensor.name == RANK_HEAD {
                tensor.format = 1;
                tensor.payload = vec![0; 12];
            }
        }
    }

    fn extra_rank_lm_head(raw: &mut RawGguf) {
        raw.tensors.push(RawTensor {
            name: "output.weight".to_string(),
            dims: vec![TEST_HIDDEN, TEST_VOCABULARY],
            format: 0,
            payload: vec![0; 48],
        });
    }

    fn extra_rank_bias(raw: &mut RawGguf) {
        raw.tensors.push(RawTensor {
            name: "cls.output.bias".to_string(),
            dims: vec![2],
            format: 0,
            payload: vec![0; 8],
        });
    }

    fn rank_causal_false(raw: &mut RawGguf) {
        raw.metadata.push(metadata_bool(CAUSAL, false));
    }

    fn mistype_hidden(raw: &mut RawGguf) {
        replace_metadata(raw, HIDDEN, &RawMetadataValue::F32(3.0));
    }

    fn causal_false(raw: &mut RawGguf) {
        raw.metadata.push(metadata_bool(CAUSAL, false));
    }

    fn unknown_rope_scaling(raw: &mut RawGguf) {
        raw.metadata
            .push(metadata_string(ROPE_SCALING_TYPE, "dynamic"));
    }

    fn nonneutral_legacy_rope_scaling(raw: &mut RawGguf) {
        raw.metadata
            .push(metadata_f32(ROPE_LEGACY_LINEAR_SCALE, 2.0));
    }

    fn nonneutral_current_rope_scaling(raw: &mut RawGguf) {
        raw.metadata.push(metadata_f32(ROPE_SCALING_FACTOR, 2.0));
    }

    fn nonneutral_rope_attention_scaling(raw: &mut RawGguf) {
        raw.metadata
            .push(metadata_f32(ROPE_SCALING_ATTENTION_FACTOR, 2.0));
    }

    fn source_neutral_none_scaling(raw: &mut RawGguf) {
        raw.metadata
            .push(metadata_string(ROPE_SCALING_TYPE, "none"));
        raw.metadata.push(metadata_f32(ROPE_SCALING_FACTOR, 2.0));
        raw.metadata
            .push(metadata_f32(ROPE_LEGACY_LINEAR_SCALE, 2.0));
    }

    fn source_neutral_current_factor_precedes_legacy(raw: &mut RawGguf) {
        raw.metadata.push(metadata_f32(ROPE_SCALING_FACTOR, 0.0));
        raw.metadata
            .push(metadata_f32(ROPE_LEGACY_LINEAR_SCALE, 2.0));
    }

    fn source_neutral_legacy_factor(raw: &mut RawGguf) {
        raw.metadata
            .push(metadata_f32(ROPE_LEGACY_LINEAR_SCALE, 1.0));
    }

    fn uneven_heads(raw: &mut RawGguf) {
        replace_metadata(raw, HEADS, &RawMetadataValue::U32(3));
        replace_metadata(raw, KV_HEADS, &RawMetadataValue::U32(2));
    }

    fn unequal_value_width(raw: &mut RawGguf) {
        replace_metadata(raw, VALUE_LENGTH, &RawMetadataValue::U32(1));
    }

    fn partial_rotary_width(raw: &mut RawGguf) {
        replace_metadata(raw, ROPE_DIMENSION, &RawMetadataValue::U32(1));
    }

    fn remove_tensor(raw: &mut RawGguf) {
        raw.tensors
            .retain(|tensor| tensor.name != "blk.0.ffn_down.weight");
    }

    fn extra_tensor(raw: &mut RawGguf) {
        raw.tensors.push(RawTensor {
            name: "output.weight".to_string(),
            dims: vec![TEST_HIDDEN, TEST_VOCABULARY],
            format: 0,
            payload: vec![0; 48],
        });
    }

    fn wrong_tensor_shape(raw: &mut RawGguf) {
        replace_tensor_dimensions(raw, "blk.0.attn_q.weight", &[TEST_HIDDEN, TEST_HEAD_DIM]);
    }

    fn wrong_matrix_dtype(raw: &mut RawGguf) {
        for tensor in &mut raw.tensors {
            if tensor.name == "blk.0.attn_q.weight" {
                tensor.format = 1;
                tensor.payload = vec![0; 24];
            }
        }
    }

    fn wrong_norm_dtype(raw: &mut RawGguf) {
        for tensor in &mut raw.tensors {
            if tensor.name == OUTPUT_NORM {
                tensor.format = 1;
                tensor.payload = vec![0; 6];
            }
        }
    }

    fn replace_metadata(raw: &mut RawGguf, key: &str, value: &RawMetadataValue) {
        for entry in &mut raw.metadata {
            if entry.key == key {
                entry.value = value.clone();
            }
        }
    }

    fn replace_tensor_dimensions(raw: &mut RawGguf, name: &str, dimensions: &[u64]) {
        for tensor in &mut raw.tensors {
            if tensor.name == name {
                tensor.dims = dimensions.to_owned();
            }
        }
    }

    fn find_tensor_mut<'raw>(
        raw: &'raw mut RawGguf,
        name: &str,
    ) -> std::result::Result<&'raw mut RawTensor, String> {
        raw.tensors
            .iter_mut()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| format!("fixture tensor {name} was absent"))
    }

    fn verify(raw: &RawGguf) -> std::result::Result<VerifiedArtifact, String> {
        let serialized = serialize_raw_gguf(raw).map_err(|error| error.to_string())?;
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        let path = directory.path().join("qwen3.gguf");
        std::fs::write(&path, &serialized.bytes).map_err(|error| error.to_string())?;
        let limit = NonZeroU64::new(serialized.byte_len + 1).ok_or("invalid fixture limit")?;
        VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(serialized.sha256),
            ArtifactByteLimit::new(limit),
        )
        .map_err(|error| error.to_string())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the fixture names every tensor in the one bounded Qwen3 profile"
    )]
    fn fixture() -> std::result::Result<RawGguf, String> {
        let mut tensors = vec![
            tensor(
                "token_embd.weight",
                vec![TEST_HIDDEN, TEST_VOCABULARY],
                0.125,
            )?,
            tensor("output_norm.weight", vec![TEST_HIDDEN], 1.0)?,
        ];
        for (role, dimensions, seed) in [
            ("attn_norm.weight", vec![TEST_HIDDEN], 1.0),
            ("attn_q_norm.weight", vec![TEST_HEAD_DIM], 1.0),
            ("attn_k_norm.weight", vec![TEST_HEAD_DIM], 1.0),
            ("ffn_norm.weight", vec![TEST_HIDDEN], 1.0),
            (
                "attn_q.weight",
                vec![TEST_HIDDEN, TEST_HEADS * TEST_HEAD_DIM],
                0.0625,
            ),
            (
                "attn_k.weight",
                vec![TEST_HIDDEN, TEST_KV_HEADS * TEST_HEAD_DIM],
                0.09375,
            ),
            (
                "attn_v.weight",
                vec![TEST_HIDDEN, TEST_KV_HEADS * TEST_HEAD_DIM],
                0.125,
            ),
            (
                "attn_output.weight",
                vec![TEST_HEADS * TEST_HEAD_DIM, TEST_HIDDEN],
                0.15625,
            ),
            (
                "ffn_gate.weight",
                vec![TEST_HIDDEN, TEST_FEED_FORWARD],
                0.1875,
            ),
            (
                "ffn_up.weight",
                vec![TEST_HIDDEN, TEST_FEED_FORWARD],
                0.21875,
            ),
            (
                "ffn_down.weight",
                vec![TEST_FEED_FORWARD, TEST_HIDDEN],
                0.25,
            ),
        ] {
            tensors.push(tensor(&format!("blk.0.{role}"), dimensions, seed)?);
        }
        Ok(RawGguf {
            metadata: vec![
                metadata_string(ARCHITECTURE, "qwen3"),
                metadata_u32(
                    BLOCK_COUNT,
                    u32::try_from(TEST_BLOCKS).map_err(|error| error.to_string())?,
                ),
                metadata_u32(CONTEXT_LENGTH, TEST_CONTEXT),
                metadata_u32(
                    HIDDEN,
                    u32::try_from(TEST_HIDDEN).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    FEED_FORWARD,
                    u32::try_from(TEST_FEED_FORWARD).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    HEADS,
                    u32::try_from(TEST_HEADS).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    KV_HEADS,
                    u32::try_from(TEST_KV_HEADS).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    KEY_LENGTH,
                    u32::try_from(TEST_HEAD_DIM).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    VALUE_LENGTH,
                    u32::try_from(TEST_HEAD_DIM).map_err(|error| error.to_string())?,
                ),
                metadata_f32(RMS_EPSILON, 0.001),
                metadata_u32(
                    ROPE_DIMENSION,
                    u32::try_from(TEST_HEAD_DIM).map_err(|error| error.to_string())?,
                ),
                metadata_f32(ROPE_BASE, 10_000.0),
                metadata_u32(
                    POOLING_TYPE,
                    u32::try_from(LAST_POOLING_TYPE).map_err(|error| error.to_string())?,
                ),
                RawMetadata {
                    key: "tokenizer.ggml.tokens".to_string(),
                    value: RawMetadataValue::StringArray(vec![
                        "alice".to_string(),
                        "bob".to_string(),
                        "acme".to_string(),
                        "corp".to_string(),
                    ]),
                },
            ],
            tensors,
        })
    }

    fn rank_fixture() -> std::result::Result<RawGguf, String> {
        let mut raw = fixture()?;
        replace_metadata(&mut raw, POOLING_TYPE, &RawMetadataValue::U32(4));
        raw.metadata.push(RawMetadata {
            key: RANK_LABELS_KEY.to_string(),
            value: RawMetadataValue::StringArray(
                RANK_LABELS
                    .iter()
                    .map(|label| (*label).to_string())
                    .collect(),
            ),
        });
        raw.tensors
            .push(tensor(RANK_HEAD, vec![TEST_HIDDEN, 2], 0.3125)?);
        Ok(raw)
    }

    fn metadata_u32(key: &str, value: u32) -> RawMetadata {
        RawMetadata {
            key: key.to_string(),
            value: RawMetadataValue::U32(value),
        }
    }
    fn metadata_f32(key: &str, value: f32) -> RawMetadata {
        RawMetadata {
            key: key.to_string(),
            value: RawMetadataValue::F32(value),
        }
    }
    fn metadata_bool(key: &str, value: bool) -> RawMetadata {
        RawMetadata {
            key: key.to_string(),
            value: RawMetadataValue::Bool(value),
        }
    }
    fn metadata_string(key: &str, value: &str) -> RawMetadata {
        RawMetadata {
            key: key.to_string(),
            value: RawMetadataValue::String(value.to_string()),
        }
    }
    fn tensor(name: &str, dims: Vec<u64>, seed: f32) -> std::result::Result<RawTensor, String> {
        let count = dims.iter().try_fold(1_usize, |count, dimension| {
            count
                .checked_mul(usize::try_from(*dimension).map_err(|error| error.to_string())?)
                .ok_or("fixture element count overflow".to_string())
        })?;
        let mut payload = Vec::with_capacity(count * 4);
        for index in 0..count {
            let index = index
                .to_f32()
                .ok_or("fixture element index cannot convert to f32")?;
            payload.extend_from_slice(&(seed + index * 0.007_812_5).to_le_bytes());
        }
        Ok(RawTensor {
            name: name.to_string(),
            dims,
            format: 0,
            payload,
        })
    }
}

#[cfg(test)]
#[path = "qwen3_oracle_tests.rs"]
mod oracle_tests;
