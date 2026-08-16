//! Device-pair placement classification shared by every two-operand op
//! (`matmul`, `rms_norm`).
//!
//! WHY a shared classifier instead of a `match` inline at each call
//! site: `matmul` and `rms_norm` each dispatched on
//! `(Option<&HipStorage>, Option<&HipStorage>)` with a two-arm match —
//! `(Some, Some)` for the GPU kernel, `_` for everything else. The
//! wildcard silently swallowed both mixed placements
//! (`(Some, None)`/`(None, Some)`) into the same host round-trip as the
//! genuine `(None, None)` case (forkwright/logismos#38, #39). Factoring
//! the three-way decision into one pure function closes both call
//! sites at once and turns the decision into something a test can pin
//! without a HIP device — it takes only presence flags, never touches
//! storage.

/// Where a two-operand op's inputs are placed, relative to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DevicePlacement {
    /// Both operands carry HIP storage — dispatch to the device kernel.
    BothHip,
    /// Neither operand carries HIP storage — dispatch to the CPU
    /// reference path.
    BothCpu,
    /// Exactly one operand carries HIP storage. A caller bug — most
    /// often a weight upload that never reached the device — and must
    /// be rejected rather than silently resolved by moving the HIP
    /// operand back to host.
    Mixed,
}

/// Classify a two-operand device pair from each operand's HIP-storage
/// presence alone. Never inspects the storage itself, so it needs no
/// HIP device and runs anywhere this crate compiles.
pub(crate) fn classify_placement(a_is_hip: bool, b_is_hip: bool) -> DevicePlacement {
    // DELIBERATE-BREAK(negative-fixture, forkwright/logismos#38/#39 PR):
    // v4 -- the literal original two-arm wildcard collapse (`_ =>`),
    // byte-for-byte the pre-fix shape, no or-pattern reshaping. v1-v3
    // each tried to keep `Mixed` reachable while still routing
    // `(true, false)` to `BothCpu`, and each was caught by clippy
    // (match_same_arms, unnested_or_patterns) before nextest ever ran
    // -- real CI rejections, but not evidence the *test* itself
    // catches this. A bare wildcard has only two match arms with two
    // distinct bodies, so neither clippy lint fires; this is the
    // version that actually reaches
    // `mixed_placement_is_neither_bothcpu_nor_bothhip` and lets it
    // assert. Not the shipped state -- reverted in the next commit.
    match (a_is_hip, b_is_hip) {
        (true, true) => DevicePlacement::BothHip,
        _ => DevicePlacement::BothCpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_hip_dispatches_to_gpu() {
        assert_eq!(classify_placement(true, true), DevicePlacement::BothHip);
    }

    #[test]
    fn both_cpu_dispatches_to_cpu() {
        assert_eq!(classify_placement(false, false), DevicePlacement::BothCpu);
    }

    #[test]
    fn mixed_placement_is_neither_bothcpu_nor_bothhip() {
        // WHY(forkwright/logismos#38, #39): the pre-fix dispatch used a
        // two-arm match whose wildcard arm routed BOTH mixed
        // combinations into the same path as `BothCpu`. This is the
        // regression test: a mixed pair must classify as its own
        // variant, distinct from `BothCpu`, so the caller can reject it
        // instead of silently transferring the HIP operand to host.
        assert_eq!(classify_placement(true, false), DevicePlacement::Mixed);
        assert_eq!(classify_placement(false, true), DevicePlacement::Mixed);
        assert_ne!(classify_placement(true, false), DevicePlacement::BothCpu);
        assert_ne!(classify_placement(false, true), DevicePlacement::BothCpu);
    }
}
