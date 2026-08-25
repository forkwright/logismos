//! Raw HIP runtime bindings.
//!
//! Generated at build time by `bindgen` from
//! `include/wrapper_runtime.h`; output lands in `$OUT_DIR/hip_bindings.rs`.
//!
//! The symbols here are FFI-unsafe in the usual way: pointers, raw
//! enums, no lifetime or thread-safety information. Every consumer
//! inside `hipcore` wraps them in a safe type (`Device`, `Stream`,
//! `DeviceBuffer`, `Event`). Nothing outside `hipcore` calls these
//! directly.

#![expect(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    clippy::pedantic,
    clippy::all,
    clippy::missing_safety_doc,
    missing_docs,
    reason = "bindgen emits raw HIP symbols that are wrapped by safe hipcore APIs"
)]

include!(concat!(env!("OUT_DIR"), "/hip_bindings.rs"));
