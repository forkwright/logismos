//! Compile-only fixture for forkwright/logismos#25.
//!
//! Never executed — copied into an ephemeral crate and type-checked by
//! `crates/hipcore/tests/copy_from_host_async_ownership.rs`, which
//! asserts this reproduction fails to compile.
//!
//! Reproduces the exact pattern from issue #25's own filed evidence:
//! start an async host -> device copy, then reuse the host buffer
//! before the copy is known to have completed. Under the pre-fix
//! `&[T]`-borrowing signature this compiled cleanly (the borrow ends
//! the instant `copy_from_host_async` returns, well before the DMA is
//! guaranteed done). Under the fixed signature `data` is moved into
//! the call, so reusing it below is a compile-time "use of moved
//! value" (E0382).

use hipcore::{Device, DeviceBuffer, Stream};

fn main() {
    let device = Device::new(0).expect("device");
    let stream = Stream::new(&device).expect("stream");
    let mut buf: DeviceBuffer<u8> = DeviceBuffer::alloc(&device, 4).expect("alloc");
    let data = vec![0_u8; 4];
    // `data` moves into `copy_from_host_async`, which owns it until the
    // returned `PendingCopy` is waited on or dropped. Reusing the
    // original binding here must not compile.
    let _pending = buf.copy_from_host_async(data, &stream);
    drop(data);
}
