//! `DeviceBuffer::copy_from_host_async` used to borrow its `data`
//! parameter as `&[T]` (forkwright/logismos#25) and its destination
//! `self` as `&mut DeviceBuffer<T>` (forkwright/logismos#104), so a
//! fully safe caller could free either buffer before the GPU's async
//! DMA finished touching it. The fix takes ownership of both instead
//! (`crates/hipcore/src/memory.rs`), so a caller who reuses either
//! buffer after handing it to the async copy gets a compile-time
//! error, not a runtime race.
//!
//! These tests prove that mechanically: each copies one of the
//! checked-in reproductions in `tests/fixtures/` (which call the
//! *real* `hipcore` crate -- not a re-implementation) into its own
//! ephemeral crate and runs `cargo check` against it, asserting the
//! build fails with E0382 ("use of moved value").
//!
//! No `ROCm` hardware exists anywhere in this fleet, so this never
//! executes GPU code. Borrow checking is a compile-time-only pass, so
//! `cargo check` proves the invariant without a device: it needs only
//! the HIP headers + `libamdhip64` link target for `hipcore` itself to
//! build, which CI provisions (`libamdhip64-dev`) and this repo's other
//! CI jobs already rely on. Each ephemeral crate is generated at
//! runtime (own temp dir, own `Cargo.lock`, own `--target-dir`) so
//! running these tests never writes build artifacts into the checked-out
//! tree.

#![expect(
    clippy::expect_used,
    reason = "test assertions use expect() directly, matching error.rs's existing test convention"
)]

use std::process::Command;

/// Copies `fixture_file` (a path under `tests/fixtures/`, relative to
/// this crate's manifest dir) into a fresh ephemeral binary crate that
/// depends on `hipcore` by path, runs `cargo check` against it, and
/// asserts the check fails with E0382 ("use of moved value") — proving
/// the fixture's post-move reuse is rejected by the shipped
/// `copy_from_host_async` signature, not merely by a re-implementation
/// of it.
fn assert_fixture_rejects_reuse_after_move(fixture_file: &str, crate_name: &str) {
    let hipcore_root = env!("CARGO_MANIFEST_DIR");
    let fixture_src = format!("{hipcore_root}/tests/fixtures/{fixture_file}");

    let work = tempfile::tempdir().expect("temp dir for the ephemeral fixture crate");
    std::fs::create_dir_all(work.path().join("src")).expect("create fixture src dir");
    std::fs::copy(&fixture_src, work.path().join("src/main.rs"))
        .expect("copy the checked-in fixture source into the ephemeral crate");
    std::fs::write(
        work.path().join("Cargo.toml"),
        format!(
            "[package]\n\
             name = {crate_name:?}\n\
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
        .arg("--offline")
        .arg("--manifest-path")
        .arg(work.path().join("Cargo.toml"))
        .arg("--target-dir")
        .arg(work.path().join("target"))
        .output()
        .expect("failed to invoke cargo against the compile-fail fixture");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "fixture {fixture_file} unexpectedly compiled -- copy_from_host_async no longer \
         rejects reuse of a value moved into the async copy:\n{stderr}"
    );
    assert!(
        stderr.contains("E0382"),
        "expected E0382 (use of moved value) rejecting the post-move reuse in \
         {fixture_file}; got a different compile failure, so this isn't proving what \
         it claims to:\n{stderr}"
    );
}

#[test]
fn copy_from_host_async_rejects_reuse_of_moved_host_buffer() {
    assert_fixture_rejects_reuse_after_move(
        "copy_from_host_async_uaf.rs",
        "copy-from-host-async-uaf-fixture",
    );
}

#[test]
fn copy_from_host_async_rejects_reuse_of_moved_destination_buffer() {
    assert_fixture_rejects_reuse_after_move(
        "copy_from_host_async_destination_uaf.rs",
        "copy-from-host-async-destination-uaf-fixture",
    );
}
