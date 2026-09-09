//! Shared classification for native creation calls with output handles.

use crate::error::Error;

/// Operations used by the single production creation driver.
///
/// Handles are constrained to `Copy`: the driver holds only inert output bits,
/// never a Rust wrapper whose destructor could call the native runtime.
pub(crate) trait CreationPort {
    type Handle: Copy;

    fn null_handle(&self) -> Self::Handle;
    fn is_null(&self, handle: Self::Handle) -> bool;
    fn preflight(&mut self) -> Result<(), Error>;
    fn create(&mut self, output: &mut Self::Handle) -> Result<(), Error>;
    fn success_with_null_error(&self) -> Error;
}

/// Classified result of exactly one native creation attempt.
pub(crate) enum CreationAttempt<H: Copy> {
    Created(H),
    NoHandle(Error),
    Indeterminate { handle: H, error: Error },
}

/// Result after an output attempt is converted into a Rust owner or quarantine.
pub(crate) enum CreationResolution<R, Q> {
    Created(R),
    NoHandle(Error),
    Quarantined(Q),
}

/// Run one preflight and, if it succeeds, exactly one creation call.
pub(crate) fn attempt_creation<P: CreationPort>(port: &mut P) -> CreationAttempt<P::Handle> {
    let mut output = port.null_handle();
    if let Err(error) = port.preflight() {
        return CreationAttempt::NoHandle(error);
    }
    match port.create(&mut output) {
        Ok(()) if port.is_null(output) => CreationAttempt::NoHandle(port.success_with_null_error()),
        Ok(()) => CreationAttempt::Created(output),
        Err(error) if port.is_null(output) => CreationAttempt::NoHandle(error),
        Err(error) => CreationAttempt::Indeterminate {
            handle: output,
            error,
        },
    }
}

/// Construct an ordinary Rust owner only after creation was acknowledged.
pub(crate) fn resolve_creation<H: Copy, R, Q>(
    attempt: CreationAttempt<H>,
    wrap_created: impl FnOnce(H) -> Result<R, Error>,
    quarantine: impl FnOnce(H, Error) -> Q,
) -> CreationResolution<R, Q> {
    match attempt {
        CreationAttempt::Created(handle) => match wrap_created(handle) {
            Ok(resource) => CreationResolution::Created(resource),
            Err(error) => CreationResolution::NoHandle(error),
        },
        CreationAttempt::NoHandle(error) => CreationResolution::NoHandle(error),
        CreationAttempt::Indeterminate { handle, error } => {
            CreationResolution::Quarantined(quarantine(handle, error))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[derive(Clone, Copy)]
    enum CreateStep {
        Success,
        Failure,
    }

    struct FakeCreationPort {
        preflight_fails: bool,
        create_step: CreateStep,
        output: usize,
        preflight_calls: usize,
        create_calls: usize,
    }

    impl FakeCreationPort {
        const fn new(preflight_fails: bool, create_step: CreateStep, output: usize) -> Self {
            Self {
                preflight_fails,
                create_step,
                output,
                preflight_calls: 0,
                create_calls: 0,
            }
        }
    }

    impl CreationPort for FakeCreationPort {
        type Handle = usize;

        fn null_handle(&self) -> Self::Handle {
            0
        }

        fn is_null(&self, handle: Self::Handle) -> bool {
            handle == 0
        }

        fn preflight(&mut self) -> Result<(), Error> {
            self.preflight_calls += 1;
            if self.preflight_fails {
                Err(failure("preflight"))
            } else {
                Ok(())
            }
        }

        fn create(&mut self, output: &mut Self::Handle) -> Result<(), Error> {
            self.create_calls += 1;
            *output = self.output;
            match self.create_step {
                CreateStep::Success => Ok(()),
                CreateStep::Failure => Err(failure("create")),
            }
        }

        fn success_with_null_error(&self) -> Error {
            failure("success-null")
        }
    }

    fn failure(operation: &'static str) -> Error {
        Error::runtime(1, operation)
    }

    #[test]
    fn preflight_failure_never_calls_creation() {
        let mut port = FakeCreationPort::new(true, CreateStep::Success, 7);
        let attempt = attempt_creation(&mut port);
        assert!(matches!(attempt, CreationAttempt::NoHandle(_)));
        assert_eq!(port.preflight_calls, 1);
        assert_eq!(port.create_calls, 0);
    }

    #[test]
    fn error_with_null_output_has_no_handle() {
        let mut port = FakeCreationPort::new(false, CreateStep::Failure, 0);
        let attempt = attempt_creation(&mut port);
        assert!(matches!(attempt, CreationAttempt::NoHandle(_)));
        assert_eq!(port.create_calls, 1);
    }

    #[test]
    fn error_with_non_null_output_is_terminal_and_inert() {
        let mut port = FakeCreationPort::new(false, CreateStep::Failure, 7);
        let attempt = attempt_creation(&mut port);
        assert!(matches!(
            attempt,
            CreationAttempt::Indeterminate { handle: 7, .. }
        ));
        assert_eq!(port.create_calls, 1);
    }

    #[test]
    fn success_with_non_null_output_creates_handle() {
        let mut port = FakeCreationPort::new(false, CreateStep::Success, 7);
        let attempt = attempt_creation(&mut port);
        assert!(matches!(attempt, CreationAttempt::Created(7)));
        assert_eq!(port.create_calls, 1);
    }

    #[test]
    fn success_with_null_output_is_rejected_without_wrapper_drop() {
        let mut port = FakeCreationPort::new(false, CreateStep::Success, 0);
        let attempt = attempt_creation(&mut port);
        assert!(matches!(attempt, CreationAttempt::NoHandle(_)));
        assert_eq!(port.create_calls, 1);
    }

    struct DropTrackedOwner {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropTrackedOwner {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct FakeQuarantine {
        _handle: usize,
        _error: Error,
    }

    #[test]
    fn indeterminate_output_never_constructs_an_ordinary_drop_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        let resolution = resolve_creation(
            CreationAttempt::Indeterminate {
                handle: 7,
                error: failure("create"),
            },
            |_| {
                Ok(DropTrackedOwner {
                    drops: Arc::clone(&drops),
                })
            },
            |handle, error| FakeQuarantine {
                _handle: handle,
                _error: error,
            },
        );
        assert!(matches!(resolution, CreationResolution::Quarantined(_)));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn acknowledged_creation_uses_the_ordinary_drop_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        let resolution = resolve_creation(
            CreationAttempt::Created(7),
            |_| {
                Ok(DropTrackedOwner {
                    drops: Arc::clone(&drops),
                })
            },
            |handle, error| FakeQuarantine {
                _handle: handle,
                _error: error,
            },
        );
        assert!(matches!(resolution, CreationResolution::Created(_)));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
