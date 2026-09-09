//! # decoders
//!
//! Decoder-only LLM family: Qwen2 / 3 (including GDN hybrid), Llama.
//!
//! The default implementation includes bounded verified-payload CPU execution
//! from Qwen3.5 token ids through main hybrid blocks to logits, retaining only
//! process-local recurrent and KV context. The optional `gpu` feature adds an
//! unsafe, blocking, one-full-block native qualification boundary; it is not
//! hybrid GPU execution, a model qualification result, or a serving surface.
//! This crate provides no tokenizer/template, sampling, `NextN`, runtime
//! admission, or artifact-parity claim.
//!
//! ## Responsibility
//!
//! - Autoregressive forward pass with KV cache
//! - Qwen2/3 architecture (GQA + `RoPE` + `SwiGLU` + `RMSNorm`)
//! - Qwen3 GDN hybrid
//! - Llama family
//!
//! This narrow path does not provide cross-session cache sharing, speculative
//! decoding, serving, or a `core::DecoderModel` admission surface.
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod error;
mod matrix;
pub mod qwen3;
pub mod qwen35;
pub mod qwen35_execution;
mod qwen35_mrope;
#[cfg(any(feature = "gpu", test))]
mod qwen35_native;
pub mod qwen35_recurrent;
mod qwen35_requirements;
pub mod qwen35_weights;
pub mod qwen3_rank;
mod qwen3_requirements;

pub use crate::error::{Error, Result};
pub use crate::qwen3::{Qwen3Execution, Qwen3Weights};
pub use crate::qwen3_rank::{Qwen3RankExecution, Qwen3RankWeights};
pub use crate::qwen3_requirements::Qwen3CpuRequirements;
pub use crate::qwen35::Qwen35StructuralProfile;
pub use crate::qwen35_execution::{Qwen35Execution, Qwen35ExecutionPlan, Qwen35LogitSelection};
#[cfg(feature = "gpu")]
pub use crate::qwen35_native::{
    Qwen35NativeExecutionDeviceDemand, Qwen35NativeExecutionModel, Qwen35NativeExecutionPlan,
    Qwen35NativeExecutionSession, Qwen35NativeExecutionSessionPlan, Qwen35NativeLayerDeviceDemand,
    Qwen35NativeLayerPlan, Qwen35NativeLayerSession, Qwen35NativeLayerSessionState,
    Qwen35NativeSessionState,
};
pub use crate::qwen35_recurrent::Qwen35RecurrentExecution;
pub use crate::qwen35_requirements::Qwen35CpuRequirements;
pub use crate::qwen35_weights::Qwen35Weights;

#[cfg(test)]
const CRATE_NAME: &str = "decoders";

#[cfg(all(test, feature = "gpu"))]
mod qwen35_native_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
