//! Gated Delta Rule kernels: HIP launcher stubs + CPU fp32 reference.
//!
//! CPU reference: `cpu.rs` — use for parity validation and headless CI.
//! HIP kernels: Phase 6a, hardware-gated (W7900 / gfx1100).
//!
//! Refs: issue #11, `research/14-gdn-aiter-preflight.md`.

pub mod cpu;
