//! SwiGLU MLP block.
//!
//! `y = down(silu(gate(x)) * up(x))`
//!
//! Qwen2 / Llama convention — no biases on any of the three projections.

use kernels::cpu_f32;

use crate::error::{Result, ShapeSnafu};

/// SwiGLU weights, HF layout `[out, in]`.
#[derive(Debug, Clone)]
pub struct SwiGluMlpWeights {
    /// `gate_proj.weight` — `[intermediate, hidden]`.
    pub w_gate: Vec<f32>,
    /// `up_proj.weight` — `[intermediate, hidden]`.
    pub w_up: Vec<f32>,
    /// `down_proj.weight` — `[hidden, intermediate]`.
    pub w_down: Vec<f32>,
}

/// SwiGLU MLP block.
#[derive(Debug, Clone)]
pub struct SwiGluMlp {
    /// Hidden size (input/output of the block).
    pub hidden: usize,
    /// Intermediate size (gate/up output width).
    pub intermediate: usize,
    /// Weights.
    pub weights: SwiGluMlpWeights,
}

impl SwiGluMlp {
    /// Construct with shape-checking.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on weight-size / config disagreement.
    pub fn new(hidden: usize, intermediate: usize, weights: SwiGluMlpWeights) -> Result<Self> {
        if weights.w_gate.len() != intermediate * hidden {
            return ShapeSnafu {
                message: format!(
                    "w_gate: expected {}, got {}",
                    intermediate * hidden,
                    weights.w_gate.len()
                ),
            }
            .fail();
        }
        if weights.w_up.len() != intermediate * hidden {
            return ShapeSnafu {
                message: format!(
                    "w_up: expected {}, got {}",
                    intermediate * hidden,
                    weights.w_up.len()
                ),
            }
            .fail();
        }
        if weights.w_down.len() != hidden * intermediate {
            return ShapeSnafu {
                message: format!(
                    "w_down: expected {}, got {}",
                    hidden * intermediate,
                    weights.w_down.len()
                ),
            }
            .fail();
        }
        Ok(Self {
            hidden,
            intermediate,
            weights,
        })
    }

    /// Forward pass over a `[seq, hidden]` input.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `x` is not a multiple of `hidden`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        if !x.len().is_multiple_of(self.hidden) {
            return ShapeSnafu {
                message: format!(
                    "mlp.forward: x.len()={} not multiple of hidden={}",
                    x.len(),
                    self.hidden
                ),
            }
            .fail();
        }
        let seq = x.len() / self.hidden;
        // gate = x @ w_gate^T   -> [seq, intermediate]
        let gate = cpu_f32::linear_t(
            x,
            &self.weights.w_gate,
            None,
            seq,
            self.intermediate,
            self.hidden,
        );
        // up = x @ w_up^T       -> [seq, intermediate]
        let up = cpu_f32::linear_t(
            x,
            &self.weights.w_up,
            None,
            seq,
            self.intermediate,
            self.hidden,
        );
        // silu(gate) .* up
        let silu_gate = cpu_f32::silu(&gate);
        let prod = cpu_f32::hadamard(&silu_gate, &up);
        // down = prod @ w_down^T -> [seq, hidden]
        let out = cpu_f32::linear_t(
            &prod,
            &self.weights.w_down,
            None,
            seq,
            self.hidden,
            self.intermediate,
        );
        Ok(out)
    }
}
