//! Owned native main-model and full-attention resource lifecycle.

#[cfg(feature = "gpu")]
mod custody;
#[cfg(feature = "gpu")]
mod dispatch;
#[cfg(feature = "gpu")]
mod finish;
#[cfg(feature = "gpu")]
mod model_plan;
#[cfg(feature = "gpu")]
mod model_resources;
#[cfg(feature = "gpu")]
mod model_session;
#[cfg(feature = "gpu")]
mod model_step;
#[cfg(feature = "gpu")]
mod plan;
#[cfg(all(test, feature = "gpu"))]
mod plan_tests;
#[cfg(feature = "gpu")]
mod recurrent;
#[cfg(feature = "gpu")]
mod recurrent_plan;
#[cfg(feature = "gpu")]
mod resources;
#[cfg(feature = "gpu")]
mod session;
#[cfg(feature = "gpu")]
mod weights;

#[cfg(feature = "gpu")]
pub use model_session::{
    Qwen35NativeExecutionDeviceDemand, Qwen35NativeExecutionModel, Qwen35NativeExecutionModelClose,
    Qwen35NativeExecutionModelTeardown, Qwen35NativeExecutionPlan, Qwen35NativeExecutionSession,
    Qwen35NativeExecutionSessionPlan, Qwen35NativeExecutionSessionTeardown,
    Qwen35NativeExecutionSessionTeardownState,
};
#[cfg(feature = "gpu")]
pub use session::{
    Qwen35NativeLayerDeviceDemand, Qwen35NativeLayerPlan, Qwen35NativeLayerSession,
    Qwen35NativeLayerSessionState, Qwen35NativeSessionState,
};

/// One owned resource bundle whose submitted work can be synchronized.
///
/// This private seam lets the native session exercise the same ownership and
/// drop path under fault injection without manufacturing a second state model.
trait CompletionResource {
    type Error;

    fn synchronize(&mut self) -> core::result::Result<(), Self::Error>;

    /// Validate synchronized device-side state before any logical publication.
    ///
    /// This has no default because every owned native resource must make an
    /// explicit decision about its post-synchronization failure boundary.
    fn validate_after_synchronization(&mut self) -> core::result::Result<(), Self::Error>;
}

enum ResourceState<Resource> {
    Ready(Resource),
    InFlight(Resource),
    PoisonedIdle(Resource),
    PoisonedUncertain(Resource),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BeginError {
    MissingResource,
    NotReady,
}

#[derive(Debug)]
enum CompletionError<Error> {
    MissingResource,
    NotSubmitted,
    Commit { source: Error },
    Synchronization { source: Error },
    PostSynchronizationValidation { source: Error },
}

struct ResourceOwner<Resource: CompletionResource> {
    state: Option<ResourceState<Resource>>,
}

impl<Resource: CompletionResource> ResourceOwner<Resource> {
    fn new(resource: Resource) -> Self {
        Self {
            state: Some(ResourceState::Ready(resource)),
        }
    }

    fn begin(&mut self) -> core::result::Result<InFlight<'_, Resource>, BeginError> {
        let state = self.state.take().ok_or(BeginError::MissingResource)?;
        let ResourceState::Ready(resource) = state else {
            self.state = Some(state);
            return Err(BeginError::NotReady);
        };
        self.state = Some(ResourceState::InFlight(resource));
        Ok(InFlight {
            owner: self,
            submitted: false,
            finished: false,
        })
    }

    fn state(&self) -> Option<&ResourceState<Resource>> {
        self.state.as_ref()
    }

    /// Consume the complete bundle without invoking its ordinary drop path.
    ///
    /// Explicit native teardown consumes this only to transfer the original
    /// stream and buffers into inert custody. `None` is possible only if an
    /// internal in-flight guard already removed the state; safe session APIs do
    /// not expose such a guard across a consuming close.
    fn into_resource(mut self) -> Option<Resource> {
        self.state.take().map(|state| match state {
            ResourceState::Ready(resource)
            | ResourceState::InFlight(resource)
            | ResourceState::PoisonedIdle(resource)
            | ResourceState::PoisonedUncertain(resource) => resource,
        })
    }
}

impl<Resource: CompletionResource> Drop for ResourceOwner<Resource> {
    fn drop(&mut self) {
        let Some(state) = self.state.take() else {
            return;
        };
        let (ResourceState::InFlight(mut resource)
        | ResourceState::PoisonedUncertain(mut resource)) = state
        else {
            return;
        };
        if resource.synchronize().is_err() {
            // A failed synchronization is not proof that device work is idle.
            // Keep the entire owned bundle alive rather than releasing buffers
            // which that work might still access.
            core::mem::forget(resource);
        }
    }
}

struct InFlight<'owner, Resource: CompletionResource> {
    owner: &'owner mut ResourceOwner<Resource>,
    submitted: bool,
    finished: bool,
}

impl<Resource: CompletionResource> InFlight<'_, Resource> {
    fn resource(
        &mut self,
    ) -> core::result::Result<&mut Resource, CompletionError<Resource::Error>> {
        match self.owner.state.as_mut() {
            Some(ResourceState::InFlight(resource)) => Ok(resource),
            Some(
                ResourceState::Ready(_)
                | ResourceState::PoisonedIdle(_)
                | ResourceState::PoisonedUncertain(_),
            )
            | None => Err(CompletionError::MissingResource),
        }
    }

    /// Marks the bundle in flight before its first device submission.
    fn mark_submitted(&mut self) {
        self.submitted = true;
    }

    /// Synchronizes submitted work, then runs the checked prepublication step.
    fn complete<Value>(
        mut self,
        commit: impl FnOnce(&mut Resource) -> core::result::Result<Value, Resource::Error>,
    ) -> core::result::Result<Value, CompletionError<Resource::Error>> {
        if !self.submitted {
            return Err(CompletionError::NotSubmitted);
        }

        let synchronization = self.resource()?.synchronize();
        if let Err(source) = synchronization {
            self.poison_uncertain();
            return Err(CompletionError::Synchronization { source });
        }

        let validation = self.resource()?.validate_after_synchronization();
        if let Err(source) = validation {
            self.poison_known_idle();
            return Err(CompletionError::PostSynchronizationValidation { source });
        }

        let value = match commit(self.resource()?) {
            Ok(value) => value,
            Err(source) => {
                self.poison_known_idle();
                return Err(CompletionError::Commit { source });
            }
        };
        self.restore_ready()?;
        Ok(value)
    }

    fn restore_ready(&mut self) -> core::result::Result<(), CompletionError<Resource::Error>> {
        let Some(ResourceState::InFlight(resource)) = self.owner.state.take() else {
            self.finished = true;
            return Err(CompletionError::MissingResource);
        };
        self.owner.state = Some(ResourceState::Ready(resource));
        self.finished = true;
        Ok(())
    }

    fn poison_uncertain(&mut self) {
        let Some(ResourceState::InFlight(resource)) = self.owner.state.take() else {
            self.finished = true;
            return;
        };
        self.owner.state = Some(ResourceState::PoisonedUncertain(resource));
        self.finished = true;
    }

    fn poison_known_idle(&mut self) {
        let Some(ResourceState::InFlight(resource)) = self.owner.state.take() else {
            self.finished = true;
            return;
        };
        self.owner.state = Some(ResourceState::PoisonedIdle(resource));
        self.finished = true;
    }

    fn finish_after_submission(&mut self) {
        let Some(ResourceState::InFlight(mut resource)) = self.owner.state.take() else {
            self.finished = true;
            return;
        };
        self.owner.state = Some(match resource.synchronize() {
            Ok(()) => ResourceState::PoisonedIdle(resource),
            Err(_) => ResourceState::PoisonedUncertain(resource),
        });
        self.finished = true;
    }
}

impl<Resource: CompletionResource> Drop for InFlight<'_, Resource> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if self.submitted {
            self.finish_after_submission();
            return;
        }
        let Some(ResourceState::InFlight(resource)) = self.owner.state.take() else {
            return;
        };
        self.owner.state = Some(ResourceState::Ready(resource));
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::panic::{AssertUnwindSafe, catch_unwind, panic_any};
    use std::rc::Rc;

    use super::{BeginError, CompletionError, CompletionResource, ResourceOwner, ResourceState};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TestError {
        ScriptFailure,
        MissingScriptEntry,
    }

    struct TestResource {
        synchronizations: Rc<Cell<usize>>,
        validations: Rc<Cell<usize>>,
        drops: Rc<Cell<usize>>,
        publications: Rc<Cell<usize>>,
        synchronization_script: VecDeque<core::result::Result<(), TestError>>,
        validation_script: VecDeque<core::result::Result<(), TestError>>,
    }

    impl TestResource {
        fn new(
            synchronizations: Rc<Cell<usize>>,
            validations: Rc<Cell<usize>>,
            drops: Rc<Cell<usize>>,
            publications: Rc<Cell<usize>>,
            synchronization_script: impl IntoIterator<Item = core::result::Result<(), TestError>>,
            validation_script: impl IntoIterator<Item = core::result::Result<(), TestError>>,
        ) -> Self {
            Self {
                synchronizations,
                validations,
                drops,
                publications,
                synchronization_script: synchronization_script.into_iter().collect(),
                validation_script: validation_script.into_iter().collect(),
            }
        }

        fn publish(&self) {
            self.publications
                .set(self.publications.get().saturating_add(1));
        }
    }

    impl CompletionResource for TestResource {
        type Error = TestError;

        fn synchronize(&mut self) -> core::result::Result<(), Self::Error> {
            self.synchronizations
                .set(self.synchronizations.get().saturating_add(1));
            match self.synchronization_script.pop_front() {
                Some(outcome) => outcome,
                None => Err(TestError::MissingScriptEntry),
            }
        }

        fn validate_after_synchronization(&mut self) -> core::result::Result<(), Self::Error> {
            self.validations
                .set(self.validations.get().saturating_add(1));
            match self.validation_script.pop_front() {
                Some(outcome) => outcome,
                None => Err(TestError::MissingScriptEntry),
            }
        }
    }

    impl Drop for TestResource {
        fn drop(&mut self) {
            self.drops.set(self.drops.get().saturating_add(1));
        }
    }

    struct OwnerFixture {
        owner: ResourceOwner<TestResource>,
        synchronizations: Rc<Cell<usize>>,
        validations: Rc<Cell<usize>>,
        drops: Rc<Cell<usize>>,
        publications: Rc<Cell<usize>>,
    }

    fn owner(
        synchronization_script: impl IntoIterator<Item = core::result::Result<(), TestError>>,
        validation_script: impl IntoIterator<Item = core::result::Result<(), TestError>>,
    ) -> OwnerFixture {
        let synchronizations = Rc::new(Cell::new(0));
        let validations = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        let publications = Rc::new(Cell::new(0));
        let owner = ResourceOwner::new(TestResource::new(
            Rc::clone(&synchronizations),
            Rc::clone(&validations),
            Rc::clone(&drops),
            Rc::clone(&publications),
            synchronization_script,
            validation_script,
        ));
        OwnerFixture {
            owner,
            synchronizations,
            validations,
            drops,
            publications,
        }
    }

    #[test]
    fn preflight_drop_returns_bundle_to_ready() -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            drops,
            ..
        } = owner([], []);
        let guard = owner.begin()?;
        drop(guard);
        assert!(matches!(owner.state(), Some(ResourceState::Ready(_))));
        assert_eq!(synchronizations.get(), 0);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn synchronized_commit_publishes_then_returns_to_ready() -> core::result::Result<(), BeginError>
    {
        let OwnerFixture {
            mut owner,
            synchronizations,
            validations,
            drops,
            publications,
        } = owner([Ok(())], [Ok(())]);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        guard
            .complete(|resource| {
                resource.publish();
                Ok(())
            })
            .map_err(|_| BeginError::MissingResource)?;
        assert!(matches!(owner.state(), Some(ResourceState::Ready(_))));
        assert_eq!(synchronizations.get(), 1);
        assert_eq!(validations.get(), 1);
        assert_eq!(publications.get(), 1);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn completion_requires_marking_before_submission() -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            drops,
            publications,
            ..
        } = owner([], []);
        let guard = owner.begin()?;
        let error = guard
            .complete(|resource| {
                resource.publish();
                Ok(())
            })
            .err()
            .ok_or(BeginError::MissingResource)?;
        assert!(matches!(error, CompletionError::NotSubmitted));
        assert!(matches!(owner.state(), Some(ResourceState::Ready(_))));
        assert_eq!(synchronizations.get(), 0);
        assert_eq!(publications.get(), 0);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn checked_prepublication_failure_is_known_idle_and_never_retries()
    -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            drops,
            publications,
            ..
        } = owner([Ok(())], [Ok(())]);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        let error = guard
            .complete(|_| Err::<(), _>(TestError::ScriptFailure))
            .err()
            .ok_or(BeginError::MissingResource)?;
        assert!(matches!(
            error,
            CompletionError::Commit {
                source: TestError::ScriptFailure,
            }
        ));
        assert!(matches!(
            owner.state(),
            Some(ResourceState::PoisonedIdle(_))
        ));
        assert!(matches!(owner.begin(), Err(BeginError::NotReady)));
        assert_eq!(publications.get(), 0);
        assert_eq!(synchronizations.get(), 1);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn submitted_drop_with_successful_sync_is_permanently_poisoned()
    -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            drops,
            publications,
            ..
        } = owner([Ok(())], []);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        drop(guard);
        assert!(matches!(
            owner.state(),
            Some(ResourceState::PoisonedIdle(_))
        ));
        assert!(matches!(owner.begin(), Err(BeginError::NotReady)));
        assert_eq!(synchronizations.get(), 1);
        assert_eq!(publications.get(), 0);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn completion_sync_failure_never_publishes_and_remains_uncertain()
    -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            drops,
            publications,
            ..
        } = owner([Err(TestError::ScriptFailure), Ok(())], []);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        let error = guard
            .complete(|resource| {
                resource.publish();
                Ok(())
            })
            .err()
            .ok_or(BeginError::MissingResource)?;
        assert!(matches!(
            error,
            CompletionError::Synchronization {
                source: TestError::ScriptFailure,
            }
        ));
        assert!(matches!(
            owner.state(),
            Some(ResourceState::PoisonedUncertain(_))
        ));
        assert_eq!(publications.get(), 0);
        assert_eq!(synchronizations.get(), 1);
        drop(owner);
        assert_eq!(synchronizations.get(), 2);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn failed_sync_drop_retries_then_forgets_entire_bundle() -> core::result::Result<(), BeginError>
    {
        let OwnerFixture {
            mut owner,
            synchronizations,
            drops,
            publications,
            ..
        } = owner(
            [Err(TestError::ScriptFailure), Err(TestError::ScriptFailure)],
            [],
        );
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        drop(guard);
        assert!(matches!(
            owner.state(),
            Some(ResourceState::PoisonedUncertain(_))
        ));
        assert_eq!(publications.get(), 0);
        drop(owner);
        assert_eq!(synchronizations.get(), 2);
        assert_eq!(drops.get(), 0);
        Ok(())
    }

    #[test]
    fn post_sync_validation_failure_never_invokes_publication()
    -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            validations,
            drops,
            publications,
        } = owner([Ok(())], [Err(TestError::ScriptFailure)]);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        let error = guard
            .complete(|resource| {
                resource.publish();
                Ok(())
            })
            .err()
            .ok_or(BeginError::MissingResource)?;
        assert!(matches!(
            error,
            CompletionError::PostSynchronizationValidation {
                source: TestError::ScriptFailure,
            }
        ));
        assert!(matches!(
            owner.state(),
            Some(ResourceState::PoisonedIdle(_))
        ));
        assert!(matches!(owner.begin(), Err(BeginError::NotReady)));
        assert_eq!(synchronizations.get(), 1);
        assert_eq!(validations.get(), 1);
        assert_eq!(publications.get(), 0);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[derive(Debug)]
    struct ControlledUnwind;

    #[test]
    #[expect(
        clippy::panic,
        reason = "test-only fault injection exercises the production guard Drop path under unwind"
    )]
    fn commit_unwind_uses_production_drop_path_and_never_publishes()
    -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            drops,
            publications,
            ..
        } = owner([Ok(()), Ok(())], [Ok(())]);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        // This controlled unwind proves the real guard's destructor synchronizes
        // and poisons the bundle if logical publication cannot finish.
        let unwind = catch_unwind(AssertUnwindSafe(|| {
            drop(guard.complete(|_| -> core::result::Result<(), TestError> {
                panic_any(ControlledUnwind)
            }));
        }));
        assert!(unwind.is_err());
        assert!(matches!(
            owner.state(),
            Some(ResourceState::PoisonedIdle(_))
        ));
        assert_eq!(publications.get(), 0);
        assert_eq!(synchronizations.get(), 2);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }
}
