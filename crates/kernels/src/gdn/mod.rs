//! Gated DeltaNet (GDN) kernel surface.
//!
//! Phase 6a: API contract + CPU stub. HIP WMMA wave32 port (gfx1100) lands in
//! Phase 6a GPU work once the preflight audit (`research/phase-6a-gdn-preflight.md`)
//! is verified against the aiter source.
//!
//! NOTE: IS_TF32_SUPPORTED is False on gfx1100; the ieee precision path is mandatory.

/// Configuration for a GDN kernel invocation.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = " Mirrors Triton kernel flag surface exactly."
)]
pub struct GdnConfig {
    /// Head dimension (must be 64 or 128).
    pub head_dim: usize,
    /// Enable gate `g` scaling in the prefill KKT path.
    pub use_g: bool,
    /// Enable per-head-key gate `gk` scaling.
    pub use_gk: bool,
    /// Enable per-head-value gate `gv` scaling.
    pub use_gv: bool,
    /// Use base-2 exponential (`exp2`) for gate decay instead of natural log.
    pub use_exp2: bool,
    /// Input uses variable-length sequences (`cu_seqlens`).
    pub is_varlen: bool,
}

/// Errors from the GDN kernel surface.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The HIP WMMA wave32 kernel is not yet implemented.
    NotImplemented,
    /// Head dimension is not supported (must be 64 or 128).
    UnsupportedHeadDim {
        /// The head dimension that was received.
        got: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(f, "GDN kernel not yet implemented"),
            Error::UnsupportedHeadDim { got } => {
                write!(f, "unsupported head dimension {got} (expected 64 or 128)")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Result alias for the GDN surface.
pub type Result<T> = core::result::Result<T, Error>;

/// Validate `config.head_dim` is supported.
///
/// WHY: gfx1100 WMMA paths are validated only for 64 and 128; other sizes
/// would need separate tile recipes.
fn check_head_dim(config: &GdnConfig) -> Result<()> {
    if config.head_dim == 64 || config.head_dim == 128 {
        Ok(())
    } else {
        Err(Error::UnsupportedHeadDim {
            got: config.head_dim,
        })
    }
}

/// CPU reference stub for `chunk_gated_delta_rule` forward pass.
///
/// NOTE: Returns `Error::NotImplemented` until the HIP WMMA port lands.
/// This stub defines the calling contract and test harness entry point.
pub fn chunk_gated_delta_rule_fwd(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _beta: &[f32],
    config: &GdnConfig,
) -> Result<Vec<f32>> {
    check_head_dim(config)?;
    Err(Error::NotImplemented)
}

/// CPU reference stub for `fused_recurrent_gated_delta_rule` forward pass.
///
/// NOTE: Returns `Error::NotImplemented` until the HIP WMMA port lands.
pub fn fused_recurrent_gated_delta_rule_fwd(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _g: Option<&[f32]>,
    config: &GdnConfig,
) -> Result<Vec<f32>> {
    check_head_dim(config)?;
    Err(Error::NotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the chunk forward stub returns `Error::NotImplemented`.
    #[test]
    fn gdn_chunk_fwd_stub_returns_not_implemented() {
        let config = GdnConfig {
            head_dim: 64,
            use_g: false,
            use_gk: false,
            use_gv: false,
            use_exp2: false,
            is_varlen: false,
        };
        let result = chunk_gated_delta_rule_fwd(&[], &[], &[], &[], &config);
        assert!(matches!(result, Err(Error::NotImplemented)));
    }

    /// Verify the fused recurrent forward stub returns `Error::NotImplemented`.
    #[test]
    fn gdn_fused_recurrent_stub_returns_not_implemented() {
        let config = GdnConfig {
            head_dim: 64,
            use_g: false,
            use_gk: false,
            use_gv: false,
            use_exp2: false,
            is_varlen: false,
        };
        let result = fused_recurrent_gated_delta_rule_fwd(&[], &[], &[], None, &config);
        assert!(matches!(result, Err(Error::NotImplemented)));
    }

    /// Verify an unsupported head dimension is rejected.
    #[test]
    fn gdn_config_rejects_bad_head_dim() {
        let config = GdnConfig {
            head_dim: 256,
            use_g: false,
            use_gk: false,
            use_gv: false,
            use_exp2: false,
            is_varlen: false,
        };
        let result = chunk_gated_delta_rule_fwd(&[], &[], &[], &[], &config);
        assert!(matches!(
            result,
            Err(Error::UnsupportedHeadDim { got: 256 })
        ));
    }

    /// Verify that 64 and 128 are accepted, then still return NotImplemented.
    #[test]
    fn gdn_config_accepts_valid_head_dims() {
        for hd in [64, 128] {
            let config = GdnConfig {
                head_dim: hd,
                use_g: false,
                use_gk: false,
                use_gv: false,
                use_exp2: false,
                is_varlen: false,
            };
            let result = chunk_gated_delta_rule_fwd(&[], &[], &[], &[], &config);
            assert!(
                matches!(result, Err(Error::NotImplemented)),
                "head_dim {hd} should be accepted but still return NotImplemented"
            );
        }
    }
}
