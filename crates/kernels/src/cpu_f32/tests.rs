//! Unit tests for the Phase-3 fp32 CPU reference kernels.

use super::*;

#[test]
fn embed_lookup_in_range_copies_rows() {
    let hidden = 2;
    let vocab = 3;
    let weight = [1.0_f32, 2.0, 10.0, 20.0, 100.0, 200.0]; // rows id0,id1,id2
    let ids = [1u32, 0, 2];
    let out = embed_lookup(&weight, hidden, vocab, &ids);
    assert_eq!(out, vec![10.0, 20.0, 1.0, 2.0, 100.0, 200.0]);
}

#[test]
fn embed_lookup_out_of_range_id_yields_zeroed_row_uniformly() {
    // WHY(forkwright/logismos#55): locks the stated contract — an
    // out-of-range token id (e.g. from a tokenizer/checkpoint
    // vocabulary mismatch) is not an error; its row is left
    // zero-filled, the same way in every build profile. Before this
    // fix the only guard was `debug_assert!(id < vocab)`, so a debug
    // build panicked on exactly this input while a release build
    // silently zero-filled — this test fails (panics) against that
    // prior behaviour and passes against the explicit runtime check.
    let hidden = 2;
    let vocab = 2;
    let weight = [1.0_f32, 2.0, 10.0, 20.0];
    let ids = [0u32, 5u32]; // id=5 is out of range for vocab=2
    let out = embed_lookup(&weight, hidden, vocab, &ids);
    assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0]);
}

#[test]
fn rms_norm_matches_manual() {
    let x = [1.0_f32, 2.0, 3.0, 4.0];
    let w = [1.0_f32; 4];
    let y = rms_norm(&x, &w, 1, 4, 1e-6);
    let inv = 1.0_f32 / f32::sqrt(((1.0 + 4.0 + 9.0 + 16.0) / 4.0) + 1e-6);
    let want = [1.0 * inv, 2.0 * inv, 3.0 * inv, 4.0 * inv];
    for (a, b) in y.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-6, "a={a} b={b}");
    }
}

#[test]
fn linear_matches_hand_computed() {
    // WHY(forkwright/logismos#44): `linear` (A @ B + bias, row-major
    // B) had no test at all, unlike its sibling `linear_t` — a
    // stride mistake here would silently corrupt every `scores @ v`
    // attention output with no panic, no assertion, no error signal.
    // m>1, n>1, k>1 so a stride/transpose bug is observable.
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
    let bias = [100.0_f32, 200.0];
    let y = linear(&a, &b, Some(&bias), 2, 2, 3);
    let expected = [104.0_f32, 205.0, 110.0, 211.0];
    for (got, want) in y.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "got {y:?}, expected {expected:?}"
        );
    }
}

#[test]
fn linear_matches_linear_t_with_transposed_weight() {
    // Cross-check: `linear_t(a, b_t, ...)` where `b_t` is `b`
    // transposed must equal `linear(a, b, ...)` — the two functions
    // encode the same product through different sgemm strides.
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
    let b_t = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 1.0]; // [2, 3], b transposed
    let via_linear = linear(&a, &b, None, 2, 2, 3);
    let via_linear_t = linear_t(&a, &b_t, None, 2, 2, 3);
    for (l, lt) in via_linear.iter().zip(via_linear_t.iter()) {
        assert!(
            (l - lt).abs() < 1e-5,
            "linear={via_linear:?}, linear_t={via_linear_t:?}"
        );
    }
}

#[test]
fn linear_t_matches_naive() {
    // x: [1, 3], w: [2, 3] → y: [1, 2]
    let x = [1.0_f32, 2.0, 3.0];
    let w = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0];
    let b = [10.0_f32, 20.0];
    let y = linear_t(&x, &w, Some(&b), 1, 2, 3);
    assert!((y[0] - (1.0 + 10.0)).abs() < 1e-6);
    assert!((y[1] - (2.0 + 3.0 + 20.0)).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "linear: a.len()=5 does not match declared shape m=2 * k=3")]
fn linear_mismatched_a_len_panics() {
    // WHY(forkwright/logismos#29): the only prior guard here was
    // `debug_assert_eq!`, which compiles to nothing under release —
    // this exact call would run `sgemm` with `a`'s 5-element buffer
    // read as if it held the 6 elements `m=2, k=3` declares, an
    // out-of-bounds read one `f32` past the end of `a` on the shipped
    // (release) code path. `#[should_panic]` fails outright if a
    // regression back to `debug_assert_eq!` ever lands, because
    // `--release -p kernels` (wired by PR #88) runs this suite with
    // debug assertions off.
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0]; // 5 elements, m*k=6 declared
    let b = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2], correct
    let _ = linear(&a, &b, None, 2, 2, 3);
}

#[test]
#[should_panic(expected = "linear: b.len()=5 does not match declared shape k=3 * n=2")]
fn linear_mismatched_b_len_panics() {
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3], correct
    let b = [1.0_f32, 0.0, 0.0, 1.0, 1.0]; // 5 elements, k*n=6 declared
    let _ = linear(&a, &b, None, 2, 2, 3);
}

#[test]
#[should_panic(expected = "linear: bias.len()=1 does not match declared n=2")]
fn linear_mismatched_bias_len_panics() {
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let bias = [1.0_f32]; // n=2 declared, 1 supplied
    let _ = linear(&a, &b, Some(&bias), 2, 2, 3);
}

#[test]
#[should_panic(expected = "linear: a.len()=2 does not match declared shape m=")]
fn linear_overflowing_shape_is_rejected_not_wrapped() {
    // WHY(forkwright/logismos#29): a naive `m * k` (plain multiplication,
    // no `checked_mul`) wraps on overflow. Picking `m` so `m * k` wraps
    // to exactly 2 and handing `linear` a real 2-element `a` would make
    // a plain `a.len() == m * k` comparison spuriously PASS — the exact
    // "shape decoded from an untrusted header" attack the issue names,
    // since a huge nonsense `m` slips past the check by wrapping to
    // match a tiny real buffer. `checked_shape_len` saturates the
    // overflowing product to `usize::MAX` instead of wrapping, so
    // `a.len()=2` correctly fails to match it and the ordinary
    // shape-mismatch `assert_eq!` fires — no real slice can ever be
    // `usize::MAX` elements long, so this path can never be bypassed by
    // a coincidental match.
    let k = 2usize;
    let m = (usize::MAX / k) + 2; // m * k wraps to 2 under wrapping_mul
    assert_eq!(
        m.wrapping_mul(k),
        2,
        "test premise: chosen m*k must wrap to 2"
    );
    let a = [1.0_f32, 2.0]; // len == the wrapped (wrong) product
    let b = [1.0_f32, 2.0]; // k=2, n=1 — irrelevant, panic fires on `a` first
    let _ = linear(&a, &b, None, m, 1, k);
}

#[test]
#[should_panic(expected = "linear: output shape m=")]
fn linear_output_shape_overflow_is_rejected_not_undersized() {
    // WHY(forkwright/logismos#29): the two read-side checks above guard
    // `a`/`b`; the WRITE side is `c`, sized by `m * n` and then written
    // into by `sgemm` at the caller's real (non-saturated) `m`/`n` via
    // raw strides. A plain `m * n` has no overflow check in a release
    // build and silently wraps, undersizing `c` while `sgemm` still
    // writes the real extent — a heap out-of-bounds WRITE, distinct
    // from (and unguarded by) the `a`/`b` mismatch checks.
    //
    // `k = 0` makes both read-side products (`m * k`, `k * n`)
    // trivially zero regardless of `m`/`n`, so `a = []`/`b = []` pass
    // their own checks for ANY `m`, `n` — isolating the write-side
    // product without needing a multi-gigabyte real allocation to
    // reach this line. `n = 2` and `m` chosen so `m * n` overflows
    // `usize` (`m.checked_mul(n)` is `None`), the same derivation
    // `linear_overflowing_shape_is_rejected_not_wrapped` above uses for
    // `m * k`.
    //
    // INVARIANT: this test only distinguishes the fix
    // (`checked_output_len`, which asserts before allocating) from the
    // defect (a raw `m * n` at the `vec!` call) under a
    // debug-assertions-off build: reverting `checked_output_len`'s call
    // site back to plain `m * n` makes Rust's own arithmetic-overflow
    // check panic instead (`attempt to multiply with overflow`, since
    // overflow-checks default on in the dev/test profile), which does
    // NOT contain the string below and so still fails this
    // `#[should_panic]` — proven pre-fix via a throwaway scratch branch
    // + CI run (see PR body).
    let k = 0usize;
    let n = 2usize;
    let m = (usize::MAX / n) + 2; // m * n overflows usize
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let _ = linear(&a, &b, None, m, n, k);
}

#[test]
#[should_panic(expected = "linear_t: output shape m=")]
fn linear_t_output_shape_overflow_is_rejected_not_undersized() {
    // Mirrors `linear_output_shape_overflow_is_rejected_not_undersized`
    // for `linear_t`'s independent `c` allocation and
    // `checked_output_len` call — `k = 0` makes `checked_shape_len(m, 0)`
    // and `checked_shape_len(n, 0)` both zero regardless of `m`/`n`, so
    // `a = []`/`b = []` pass trivially and only the write-side `m * n`
    // product is exercised.
    let k = 0usize;
    let n = 2usize;
    let m = (usize::MAX / n) + 2;
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let _ = linear_t(&a, &b, None, m, n, k);
}

#[test]
#[should_panic(expected = "linear_t: a.len()=2 does not match declared shape m=1 * k=3")]
fn linear_t_mismatched_a_len_panics() {
    let a = [1.0_f32, 2.0]; // 2 elements, m*k=3 declared
    let w = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0]; // [2, 3], correct
    let _ = linear_t(&a, &w, None, 1, 2, 3);
}

#[test]
#[should_panic(expected = "linear_t: b.len()=5 does not match declared shape n=2 * k=3")]
fn linear_t_mismatched_b_len_panics() {
    let a = [1.0_f32, 2.0, 3.0];
    let w = [1.0_f32, 0.0, 0.0, 0.0, 1.0]; // 5 elements, n*k=6 declared
    let _ = linear_t(&a, &w, None, 1, 2, 3);
}

#[test]
fn softmax_rows_sum_to_one() {
    let x = [0.0_f32, 1.0, 2.0, -1.0, 0.0, 1.0];
    let y = softmax_last_dim(&x, 2, 3);
    for r in 0..2 {
        let s: f32 = y[r * 3..(r + 1) * 3].iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}

#[test]
fn softmax_fully_masked_row_is_uniform_not_nan() {
    // WHY(forkwright/logismos#30): a row that is entirely
    // `f32::NEG_INFINITY` (a fully-masked attention row) used to
    // produce `NaN` in every slot via `(NEG_INF - NEG_INF).exp()`.
    // It must instead be a finite, uniform distribution.
    let x = [f32::NEG_INFINITY; 4];
    let y = softmax_last_dim(&x, 1, 4);
    assert!(y.iter().all(|v| v.is_finite()), "row contains NaN: {y:?}");
    let sum: f32 = y.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "row does not sum to 1: {sum}");
    for v in &y {
        assert!((v - 0.25).abs() < 1e-6, "row is not uniform: {y:?}");
    }
}

#[test]
fn softmax_mixed_masked_and_unmasked_rows_both_finite() {
    // A batch where one row is fully masked and the other is not —
    // the fully-masked row must not poison the unmasked one, and both
    // must come back finite.
    let x = [
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        0.0,
        1.0,
        2.0,
    ];
    let y = softmax_last_dim(&x, 2, 3);
    assert!(
        y.iter().all(|v| v.is_finite()),
        "output contains NaN: {y:?}"
    );
    let row0_sum: f32 = y[0..3].iter().sum();
    let row1_sum: f32 = y[3..6].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-6);
    assert!((row1_sum - 1.0).abs() < 1e-6);
}

#[test]
fn rope_zero_pos_identity_halves() {
    let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let (cos, sin) = build_rope_table_f32(1, 4, 1_000_000.0);
    rope_halves_in_place(&mut x, &cos, &sin, 1, 4);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rope_table_nonzero_position_matches_hand_computed_angles() {
    // WHY(forkwright/logismos#45): the only prior test used seq=1, so
    // every emitted angle was `0 * inv_freq = 0` and every (cos, sin)
    // pair was trivially (1, 0) regardless of whether the inv_freq /
    // powf math was even correct. This checks pos=1 against hand
    // computed values at two different `i` (so `inv_freq`'s exponent
    // is exercised at both 0 and non-zero) — a sign error in the
    // `powf` exponent or an off-by-one in `inv_freq` fails this.
    let head_dim = 4;
    let half = head_dim / 2;
    let theta = 10_000.0;
    let (cos, sin) = build_rope_table_f32(2, head_dim, theta);

    // pos=1, i=0: inv_freq = theta^0 = 1.0, angle = 1.0.
    let expected_cos_i0 = 1.0_f64.cos();
    let expected_sin_i0 = 1.0_f64.sin();
    assert!(
        (f64::from(cos[half]) - expected_cos_i0).abs() < 1e-5,
        "cos[pos=1,i=0] = {}, expected {expected_cos_i0}",
        cos[half]
    );
    assert!(
        (f64::from(sin[half]) - expected_sin_i0).abs() < 1e-5,
        "sin[pos=1,i=0] = {}, expected {expected_sin_i0}",
        sin[half]
    );

    // pos=1, i=1: inv_freq = theta^(-2*1/4) = theta^-0.5 = 0.01,
    // angle = 1.0 * 0.01 = 0.01.
    let expected_cos_i1 = 0.01_f64.cos();
    let expected_sin_i1 = 0.01_f64.sin();
    assert!(
        (f64::from(cos[half + 1]) - expected_cos_i1).abs() < 1e-5,
        "cos[pos=1,i=1] = {}, expected {expected_cos_i1}",
        cos[half + 1]
    );
    assert!(
        (f64::from(sin[half + 1]) - expected_sin_i1).abs() < 1e-5,
        "sin[pos=1,i=1] = {}, expected {expected_sin_i1}",
        sin[half + 1]
    );
}

#[test]
fn rope_rotation_is_inverse_after_pi() {
    // After a full 2π rotation the vector returns; use theta=1 so angle
    // grows fast. Direct check: rotate forward, rotate backward → identity.
    let mut x = vec![1.0_f32, 2.0];
    let cos = vec![0.5_f32.sqrt()];
    let sin = vec![0.5_f32.sqrt()];
    let orig = x.clone();
    rope_halves_in_place(&mut x, &cos, &sin, 1, 2);
    // Inverse rotation: swap sign of sin.
    let inv_sin = vec![-0.5_f32.sqrt()];
    rope_halves_in_place(&mut x, &cos, &inv_sin, 1, 2);
    for (a, b) in x.iter().zip(orig.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn mean_pool_respects_mask() {
    let h = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mask = [1_u8, 0, 1];
    let pooled = mean_pool_masked(&h, &mask, 3, 2);
    assert_eq!(pooled, vec![3.0, 4.0]);
}

#[test]
fn mean_pool_masked_all_zero_mask_stays_finite() {
    // WHY(forkwright/logismos#59): the only prior test
    // (`mean_pool_respects_mask`) uses mask [1,0,1] (den=2), never
    // exercising the `den == 0` fallback (`inv = 1.0`). Without
    // that guard an all-masked row divides by zero and every
    // output slot becomes NaN — this is the negative-case fixture
    // for that guard: it fails (NaN) if the `den > 0.0` check is
    // ever dropped.
    let h = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mask = [0_u8, 0, 0];
    let pooled = mean_pool_masked(&h, &mask, 3, 2);
    assert!(
        pooled.iter().all(|v| v.is_finite()),
        "all-masked pool must stay finite, got {pooled:?}"
    );
    assert_eq!(pooled, vec![0.0, 0.0]);
}

#[test]
fn l2_normalize_projects_to_unit() {
    let mut v = vec![3.0_f32, 4.0];
    l2_normalize_in_place(&mut v);
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
}

#[test]
fn mask_additive_broadcasts_across_heads() {
    let mut s = vec![1.0_f32; 12]; // 4 rows, 3 cols
    let mask = [1_u8, 0, 1, 1, 1, 1]; // 2 mask rows
    mask_additive_in_place(&mut s, &mask, 4, 3);
    // row 0 + 1 see mask row 0 [1,0,1]; row 2+3 see mask row 1 [1,1,1]
    assert!(s[0].is_finite());
    assert!(s[1].is_infinite() && s[1].is_sign_negative());
    assert!(s[2].is_finite());
    assert!(s[4].is_infinite() && s[4].is_sign_negative());
    assert!(s[6..].iter().all(|v| v.is_finite()));
}

#[test]
fn mask_additive_zero_n_does_not_panic() {
    // n == 0: mask.len() / n would divide by zero pre-fix.
    let mut s: Vec<f32> = vec![];
    let mask: [u8; 0] = [];
    mask_additive_in_place(&mut s, &mask, 4, 0);
    assert!(
        s.is_empty(),
        "n == 0 must stay a no-op on an empty score buffer"
    );
}

#[test]
#[should_panic(expected = "not a multiple of mask_rows")]
fn mask_additive_non_dividing_rows_panics() {
    // WHY(forkwright/logismos#59): rows=5 is not a multiple of
    // mask_rows=2 (mask.len()=6, n=3 -> mask_rows=2). Before this
    // fix, `debug_assert!` was stripped in release and `repeat =
    // rows / mask_rows` floor-divided, silently leaving the
    // trailing row unmasked instead of erroring.
    //
    // INVARIANT: this test only distinguishes the fix (`assert!`) from
    // the defect (`debug_assert!`) under a debug-assertions-off build —
    // both panic identically otherwise. `.github/workflows/gate-attestation.yml`
    // runs this crate a second time under `--release` for exactly that
    // reason; a `cargo nextest run --workspace` alone (debug-assertions
    // on by default) cannot tell the two apart and would pass unchanged
    // if this were reverted to `debug_assert!`.
    let mut s = vec![1.0_f32; 15]; // 5 rows, 3 cols
    let mask = [1_u8, 0, 1, 1, 1, 1]; // 2 mask rows
    mask_additive_in_place(&mut s, &mask, 5, 3);
}

#[test]
fn mask_additive_empty_mask_does_not_panic() {
    // rows > 0 with an empty mask: mask_rows == 0, so rows / mask_rows
    // would divide by zero pre-fix.
    let mut s = vec![1.0_f32; 12]; // 4 rows, 3 cols
    let mask: [u8; 0] = [];
    mask_additive_in_place(&mut s, &mask, 4, 3);
    // No-op: scores are untouched since there is no mask to apply.
    // WHY bit-pattern and not `==`: "untouched" is a bytes question, not an
    // approximate one. Comparing bits is stricter than `==` here — it also
    // catches a write of `-0.0`, which compares equal to `0.0` — and it is
    // what `clippy::float_cmp` is steering away from exact `==` toward.
    let untouched = 1.0_f32.to_bits();
    assert!(s.iter().all(|&v| v.to_bits() == untouched));
}

#[test]
#[should_panic(expected = "mismatched operand lengths")]
fn hadamard_mismatched_lengths_panics() {
    // WHY(forkwright/logismos#59): before this fix,
    // `debug_assert_eq!` was stripped in release and `zip` silently
    // truncated to the shorter operand. `hadamard` is a reference
    // kernel other GPU kernels are validated against, so a
    // silently-shorter output can make a parity test pass while
    // comparing against wrong data.
    //
    // INVARIANT: this test only distinguishes the fix (`assert_eq!`)
    // from the defect (`debug_assert_eq!`) under a debug-assertions-off
    // build — both panic identically otherwise. `.github/workflows/gate-attestation.yml`
    // runs this crate a second time under `--release` for exactly that
    // reason; a `cargo nextest run --workspace` alone (debug-assertions
    // on by default) cannot tell the two apart and would pass unchanged
    // if this were reverted to `debug_assert_eq!`.
    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 2.0];
    let _ = hadamard(&a, &b);
}

#[test]
fn silu_matches_manual() {
    let y = silu(&[0.0, 1.0, -1.0]);
    assert!((y[0]).abs() < 1e-6);
    assert!((y[1] - 1.0 / (1.0 + (-1.0_f32).exp())).abs() < 1e-6);
    assert!((y[2] - (-1.0) / (1.0 + 1.0_f32.exp())).abs() < 1e-6);
}
