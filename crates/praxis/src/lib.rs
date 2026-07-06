//! # praxis — πρᾶξις
//!
//! Composed high-level ops. Every entry point consumes `taxis::Tensor`
//! and dispatches through `kernels` (for device execution) or through
//! the CPU reference path. Free functions only — no backend trait,
//! no builder.
//!
//! Phase 1 exposes:
//!
//! - [`matmul`] — `D = A @ B`, fp16 in, fp16 out, fp32 accumulate.
//! - [`rms_norm`] — per-row RMSNorm.
//! - [`softmax`] — row-wise softmax along the last axis.
//! - [`rope_apply`] — rotary embedding, producing a fresh tensor.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::single_match_else
)]

pub mod error;
mod matmul;
mod norm;
mod rope;
mod softmax;

pub use crate::error::{Error, Result};
pub use crate::matmul::matmul;
pub use crate::norm::rms_norm;
pub use crate::rope::{CosSinTable, rope_apply};
pub use crate::softmax::softmax;
