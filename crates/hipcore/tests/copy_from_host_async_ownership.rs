//! Negative-case fixture for forkwright/logismos#25.
//!
//! `DeviceBuffer::copy_from_host_async` used to borrow its `data`
//! parameter as `&[T]`, so a fully safe caller could free the host
//! buffer before the GPU's async DMA finished reading it -- a
//! use-after-free reachable with no `unsafe`. The fix takes ownership
//! of `data` instead (`crates/hipcore/src/memory.rs`), so a caller who
//! reuses the buffer after handing it to the async copy gets a
//! compile-time error, not a runtime race.
//!
//! This test proves that mechanically: it copies the checked-in
//! reproduction (`tests/fixtures/copy_from_host_async_uaf.rs`, which
//! calls the *real* `hipcore` crate -- not a re-implementation) into an
//! ephemeral crate and runs `cargo check` against it, asserting the
//! build fails with E0382 ("use of moved value").
//!
//! No `ROCm` hardware exists anywhere in this fleet, so this never
//! executes GPU code. Borrow checking is a compile-time-only pass, so
//! `cargo check` proves the invariant without a device: it needs only
//! the HIP headers + `libamdhip64` link target for `hipcore` itself to
//! build, which CI provisions (`libamdhip64-dev`) and this repo's other
//! CI jobs already rely on. The ephemeral crate is generated at runtime
//! (own temp dir, own `Cargo.lock`, own `--target-dir`) so running this
//! test never writes build artifacts into the checked-out tree.

#![expect(
    clippy::expect_used,
    reason = "test assertions use expect() directly, matching error.rs's existing test convention"
)]

use std::process::Command;

#[test]
fn copy_from_host_async_rejects_reuse_of_moved_host_buffer() {
    let hipcore_root = env!("CARGO_MANIFEST_DIR");
    let fixture_src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/copy_from_host_async_uaf.rs"
    );

    let work = tempfile::tempdir().expect("temp dir for the ephemeral fixture crate");
    std::fs::create_dir_all(work.path().join("src")).expect("create fixture src dir");
    std::fs::copy(fixture_src, work.path().join("src/main.rs"))
        .expect("copy the checked-in fixture source into the ephemeral crate");
    std::fs::write(
        work.path().join("Cargo.toml"),
        format!(
            "[package]\n\
             name = \"copy-from-host-async-uaf-fixture\"\n\
             version = \"0.0.0\"\n\
             edition = \"2024\"\n\
             publish = false\n\
             \n\
             [dependencies]\n\
             hipcore = {{ path = {hipcore_root:?} }}\n\
             \n\
             [workspace]\n"
        ),
    )
    .expect("write the ephemeral fixture manifest");

    // WHY: a bare `cargo` invocation without target-dir discipline is a
    // basanos violation (`WORKFLOW/cargo-without-target-dir-in-worktree`)
    // -- pointed at a dir inside `work` so it is cleaned up with it and
    // never lands inside the checked-out tree.
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .arg("check")
        .arg("--manifest-path")
        .arg(work.path().join("Cargo.toml"))
        .arg("--target-dir")
        .arg(work.path().join("target"))
        .output()
        .expect("failed to invoke cargo against the compile-fail fixture");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "fixture unexpectedly compiled -- copy_from_host_async no longer rejects \
         reuse of `data` after it has been moved into the async copy, which reopens \
         forkwright/logismos#25:\n{stderr}"
    );
    assert!(
        stderr.contains("E0382"),
        "expected E0382 (use of moved value) rejecting the post-move reuse of \
         `data`; got a different compile failure, so this isn't proving what it \
         claims to:\n{stderr}"
    );
}
