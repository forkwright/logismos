//! Compile-only fixture for forkwright/logismos#104.
//!
//! Never executed — copied into an ephemeral crate and type-checked by
//! `crates/hipcore/tests/copy_from_host_async_ownership.rs`, which
//! asserts this reproduction fails to compile.
//!
//! Reproduces the exact pattern from issue #104's own filed evidence:
//! start an async host -> device copy, then reuse the DESTINATION
//! buffer before the copy is known to have completed. Under the
//! pre-fix `&mut self`-borrowing signature this compiled cleanly (the
//! borrow lasted only for the duration of the call, well before the
//! DMA is guaranteed done, so `drop(buf)` right after was fully safe
//! Rust — the mirror image of #25, on the destination instead of the
//! source). Under the fixed signature `self` is moved into the call
//! exactly as `data` already was, so reusing `buf` below is a
//! compile-time "use of moved value" (E0382) — the same class of error
//! the sibling `copy_from_host_async_uaf.rs` fixture already proves for
//! the source side.

use hipcore::{Device, DeviceBuffer, Stream};

fn main() {
    let device = Device::new(0).expect("device");
    let stream = Stream::new(&device).expect("stream");
    let buf: DeviceBuffer<u8> = DeviceBuffer::alloc(&device, 4).expect("alloc");
    let data = vec![0_u8; 4];
    // `buf` moves into `copy_from_host_async`, which owns it (via the
    // returned `PendingCopy`) until the copy completes. Reusing the
    // original binding here must not compile.
    let _pending = buf.copy_from_host_async(data, &stream);
    drop(buf);
}
