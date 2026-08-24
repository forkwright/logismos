//! # taxis — τάξις
//!
//! Typed tensors: shape, stride, dtype, device placement. The data
//! vessel of the platform.
//!
//! Runtime dtype + runtime rank, cheap `Arc` clone, `Arc<Storage>`
//! backing. No const generics, no backend trait. Two storage variants
//! in Phase 1 — CPU and HIP. Every HIP storage-backed tensor pins a
//! `hipcore::Device` for kernel dispatch.
//!
//! Design source: `~/dev/kanon/projects/logismos/research/02-rust-frameworks.md`
//! §3 (candle-shaped tensor) +
//! `~/dev/kanon/projects/logismos/phases/01-foundation/PLAN.md` §6.2.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::return_self_not_must_use,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::missing_errors_doc
)]

pub mod dtype;
pub mod error;
pub mod layout;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use crate::dtype::{DType, DTyped};
pub use crate::error::{Error, Result};
pub use crate::layout::Layout;
pub use crate::shape::Shape;
pub use crate::storage::{CpuStorage, HipStorage, Storage};
pub use crate::tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_byte_count_rounds_subbyte_storage() {
        assert_eq!(DType::I4.byte_count(3), 2);
    }
}
