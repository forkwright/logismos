//! # kernels
//!
//! HIP kernel launchers + CPU reference implementations. Every kernel
//! that targets the GPU has a CPU reference in the same module; the
//! parity test harness gates both paths at 1e-3 relative tolerance
//! (1e-2 for bf16) before a kernel is considered correct.
//!
//! Kernels land one-per-op in their own module tree:
//!
//! ```text
//! kernels/src/
//!     matmul/
//!         mod.rs      — Rust launcher + FFI
//!         cpu.rs      — reference implementation
//!         hip/*.hip   — device source
//!         hip/*.cpp   — extern "C" launcher shim
//!     rms_norm/ …
//!     softmax/ …
//!     rope/ …
//! ```

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::return_self_not_must_use,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::missing_errors_doc,
    clippy::missing_safety_doc
)]

pub mod cpu_f32;
pub mod error;
pub mod gdn;
pub mod matmul;
pub mod rms_norm;
pub mod rope;
pub mod softmax;

pub use crate::error::{Error, Result};
pub use crate::gdn::{GdnError, GdnResult, RecurrentInput, RecurrentOutput, recurrent_fwd};
