//! Single-slot pooled HIP stream, reused across calls on the same
//! device instead of creating and destroying one per kernel launch.

use std::ffi::c_int;
use std::sync::{Mutex, PoisonError};

use hipcore::{Device, Stream};

use crate::error::{Error, Result};

/// Pooled stream, keyed by device ordinal.
///
/// Every `praxis` op that launches a kernel synchronizes its stream
/// before returning, so callers never observe outstanding work on the
/// pooled stream — reuse across ops (and across calls to the same op)
/// is safe as long as access stays serialized. `Stream` is
/// deliberately not `Sync` (see `hipcore::Stream`'s own doc comment:
/// "concurrent use from two threads is undefined"); the `Mutex` is
/// what makes serialized access a proven property of this pool rather
/// than an assumption about how callers happen to use it.
pub(crate) struct StreamPool {
    slot: Mutex<Option<(c_int, Stream)>>,
}

impl StreamPool {
    pub(crate) const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// Run `f` against the pooled stream for `device`, creating one
    /// first if the pool is empty or holds a different device's stream.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] if a fresh stream must be created and creation
    /// fails; propagates whatever `f` returns otherwise.
    pub(crate) fn with_stream<R>(
        &self,
        device: &Device,
        f: impl FnOnce(&Stream) -> Result<R>,
    ) -> Result<R> {
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if !matches!(&*slot, Some((ordinal, _)) if *ordinal == device.ordinal()) {
            *slot = Some((device.ordinal(), Stream::new(device)?));
        }
        let Some((_, stream)) = slot.as_ref() else {
            // WHY: unreachable in practice — the branch above always
            // populates `slot` first. A typed error rather than a panic
            // because a defensive branch is not the place to introduce
            // the one `unwrap`-shaped failure mode this module exists
            // to avoid.
            return Err(Error::Invalid {
                op: "pooled_stream",
                msg: "stream pool slot unexpectedly empty after population".into(),
            });
        };
        f(stream)
    }
}

/// Crate-wide pool shared by every op. Ops synchronize before
/// returning (see [`StreamPool`]'s doc comment), so sharing one slot
/// across `matmul`, `norm`, `rope`, and `softmax` is safe and
/// maximizes reuse rather than fragmenting the pool per op.
pub(crate) static POOL: StreamPool = StreamPool::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_starts_empty() {
        let pool = StreamPool::new();
        assert!(
            pool.slot
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_none()
        );
    }
}
