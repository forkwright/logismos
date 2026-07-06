//! # diffusion
//!
//! Image-model backend ops: U-Net / `DiT` forward, schedulers, VAE,
//! CLIP + T5 text encoders. `ComfyUI` stays as the workflow-authoring
//! surface; logismos owns the inference compute.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - SDXL U-Net forward + VAE encode/decode
//! - FLUX `DiT` forward with FP8-mixed weights
//! - Schedulers (DDIM, Euler, DPM++)
//! - CLIP-L + T5-XXL text encoders
//!
//! Lands in Phase 11 (`ComfyUI` backend cutover).
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "diffusion";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
