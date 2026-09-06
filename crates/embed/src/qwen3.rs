//! Bounded CPU Qwen3 retrieval embeddings.

use decoders::Qwen3Weights;
use loader::gguf::{MetaValue, VerifiedArtifact};
use logismos_core::{EmbeddingError, EmbeddingModel, EncodeOpts, Prompt};
use snafu::{ResultExt, ensure};
use tokenize::VerifiedTokenizer;

use crate::error::{DecodersSnafu, InputTooLongSnafu, NonNormalizableSnafu, Result, TokenizeSnafu, UnresolvedPromptRoleSnafu};

const TOKENS: &str = "tokenizer.ggml.tokens";
const BOS: &str = "tokenizer.ggml.bos_token_id";
const EOS: &str = "tokenizer.ggml.eos_token_id";
const ADD_BOS: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS: &str = "tokenizer.ggml.add_eos_token";

/// Trusted instructions for semantic retrieval roles.
#[derive(Clone, Debug, Default)]
pub struct Qwen3RolePrefixes { pub s2s_query: Option<String>, pub s2p_query: Option<String> }

/// Explicit CPU request limits for Qwen3 embeddings.
#[derive(Clone, Copy, Debug)]
pub struct Qwen3EmbeddingLimits { pub max_text_bytes: usize, pub max_tokens: usize }

/// Artifact-bound CPU Qwen3 embedding model with last-token pooling.
pub struct Qwen3EmbeddingModel<'artifact> {
    weights: Qwen3Weights<'artifact>, tokenizer: VerifiedTokenizer, hidden: usize,
    max_tokens: usize, prefixes: Qwen3RolePrefixes, bos: Option<u32>, eos: Option<u32>,
    add_bos: bool, add_eos: bool, supported: [usize; 1],
}

impl<'artifact> Qwen3EmbeddingModel<'artifact> {
    /// Construct a bounded CPU-only Qwen3 embedding adapter from trusted setup.
    pub fn from_verified_cpu(artifact: &'artifact VerifiedArtifact, tokenizer: VerifiedTokenizer, limits: Qwen3EmbeddingLimits, prefixes: Qwen3RolePrefixes) -> Result<Self> {
        let metadata = artifact.observation().metadata();
        let values = match metadata.get(TOKENS) { Some(MetaValue::Array(values)) => values.values(), _ => return crate::error::LoaderSnafu { source: loader::Error::Gguf { offset: 0, msg: "missing tokenizer.ggml.tokens".into(), location: snafu::Location::caller() } }.fail() };
        let expected = values.iter().map(|value| match value { MetaValue::String(value) => value.as_str(), _ => "" });
        tokenizer.verify_exact_vocabulary(values.len(), expected).context(TokenizeSnafu)?;
        let add_bos = flag(metadata, ADD_BOS)?; let add_eos = flag(metadata, ADD_EOS)?;
        let bos = id(metadata, BOS)?; let eos = id(metadata, EOS)?;
        if add_bos { tokenizer.verify_declared_special_id(values.len(), bos.ok_or_else(|| crate::error::InputTooLongSnafu { got: 0, limit: 0 }.build())?).context(TokenizeSnafu)?; }
        if add_eos { tokenizer.verify_declared_special_id(values.len(), eos.ok_or_else(|| crate::error::InputTooLongSnafu { got: 0, limit: 0 }.build())?).context(TokenizeSnafu)?; }
        let weights = Qwen3Weights::try_from_verified(artifact).context(DecodersSnafu)?;
        let hidden = weights.hidden_width();
        Ok(Self { weights, tokenizer, hidden, max_tokens: limits.max_tokens.min(weights.max_context()), prefixes, bos, eos, add_bos, add_eos, supported: [hidden] })
    }
    /// Encode with detailed native errors.
    pub fn encode_cpu(&self, text: &str, opts: &EncodeOpts) -> Result<Vec<f32>> {
        if text.len() > self.max_tokens { return InputTooLongSnafu { got: text.len(), limit: self.max_tokens }.fail(); }
        let mut text = prefix(&self.prefixes, opts.prompt.as_ref(), text)?;
        let mut ids = self.tokenizer.tokenizer().encode(&text, false).context(TokenizeSnafu)?;
        if self.add_bos { ids.insert(0, self.bos.ok_or_else(|| UnresolvedPromptRoleSnafu { role: "BOS" }.build())?); }
        if self.add_eos { ids.push(self.eos.ok_or_else(|| UnresolvedPromptRoleSnafu { role: "EOS" }.build())?); }
        let limit = opts.max_tokens.unwrap_or(self.max_tokens);
        if ids.is_empty() || ids.len() > limit { return InputTooLongSnafu { got: ids.len(), limit }.fail(); }
        let hidden = self.weights.execution(limit).context(DecodersSnafu)?.last_hidden(&ids).context(DecodersSnafu)?;
        normalize(hidden)
    }
}

impl EmbeddingModel for Qwen3EmbeddingModel<'_> {
 fn default_dim(&self)->usize { self.hidden } fn supported_dims(&self)->&[usize] { &self.supported } fn max_tokens(&self)->usize { self.max_tokens }
 fn encode(&self,text:&str,opts:&EncodeOpts)->std::result::Result<Vec<f32>,EmbeddingError> { if opts.dim.is_some_and(|dim| dim!=self.hidden) { return Err(EmbeddingError::UnsupportedDim { dim: opts.dim.unwrap(), location:snafu::Location::caller() }); } self.encode_cpu(text,opts).map_err(|error| EmbeddingError::Compute { message:error.to_string(), location:snafu::Location::caller() }) }
}
fn flag(metadata:&std::collections::HashMap<String,MetaValue>,key:&str)->Result<bool>{Ok(matches!(metadata.get(key),Some(MetaValue::Bool(true))))}
fn id(metadata:&std::collections::HashMap<String,MetaValue>,key:&str)->Result<Option<u32>>{match metadata.get(key){None=>Ok(None),Some(MetaValue::U32(value))=>Ok(Some(*value)),_=>crate::error::InputTooLongSnafu{got:0,limit:0}.fail()}}
fn prefix(prefixes:&Qwen3RolePrefixes,prompt:Option<&Prompt>,text:&str)->Result<String>{match prompt {None=>Ok(text.to_owned()),Some(Prompt::Custom(prefix))=>Ok(format!("{prefix}{text}")),Some(Prompt::S2sQuery)=>prefixes.s2s_query.as_ref().map(|prefix|format!("{prefix}{text}")).ok_or_else(||UnresolvedPromptRoleSnafu{role:"S2sQuery"}.build()),Some(Prompt::S2pQuery)=>prefixes.s2p_query.as_ref().map(|prefix|format!("{prefix}{text}")).ok_or_else(||UnresolvedPromptRoleSnafu{role:"S2pQuery"}.build()),Some(_)=>UnresolvedPromptRoleSnafu{role:"unknown"}.fail()}}
fn normalize(mut values:Vec<f32>)->Result<Vec<f32>>{let norm=values.iter().map(|value|value*value).sum::<f32>().sqrt();if !norm.is_finite()||norm==0.0||values.iter().any(|value|!value.is_finite()){return NonNormalizableSnafu.fail();}for value in &mut values {*value/=norm;}Ok(values)}
