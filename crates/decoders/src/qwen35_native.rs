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
    NativeBuildFailure, NativeBuildRelease, NativeBuildReleaseState, NativeBuildSource,
    Qwen35NativeExecutionBatchDeviceDemand, Qwen35NativeExecutionBatchPlan,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResourceState {
    Ready,
    InFlight,
    PoisonedIdle,
    PoisonedUncertain,
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
    state: ResourceState,
    resource: Option<Resource>,
}

impl<Resource: CompletionResource> ResourceOwner<Resource> {
    fn new(resource: Resource) -> Self {
        Self {
            state: ResourceState::Ready,
            resource: Some(resource),
        }
    }

    fn begin(&mut self) -> core::result::Result<InFlight<'_, Resource>, BeginError> {
        if self.resource.is_none() {
            return Err(BeginError::MissingResource);
        }
        if self.state != ResourceState::Ready {
            return Err(BeginError::NotReady);
        }
        self.state = ResourceState::InFlight;
        Ok(InFlight {
            owner: Some(self),
            submitted: false,
            finished: false,
        })
    }

    fn state(&self) -> ResourceState {
        self.state
    }

    fn ready_resource(&self) -> core::result::Result<&Resource, BeginError> {
        if self.state != ResourceState::Ready {
            return Err(BeginError::NotReady);
        }
        self.resource.as_ref().ok_or(BeginError::MissingResource)
    }

    /// Consume the complete bundle without invoking its ordinary drop path.
    ///
    /// Explicit native teardown consumes this only to transfer the original
    /// stream and buffers into inert custody. `None` is possible only if an
    /// internal guard has already detached its resource; safe session APIs do
    /// not expose such a guard across a consuming close.
    fn into_resource(mut self) -> Option<Resource> {
        self.resource.take()
    }
}

impl<Resource: CompletionResource> Drop for ResourceOwner<Resource> {
    fn drop(&mut self) {
        if !matches!(
            self.state,
            ResourceState::InFlight | ResourceState::PoisonedUncertain
        ) {
            return;
        }
        let Some(mut resource) = self.resource.take() else {
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
    owner: Option<&'owner mut ResourceOwner<Resource>>,
    submitted: bool,
    finished: bool,
}

impl<'owner, Resource: CompletionResource> InFlight<'owner, Resource> {
    fn owner_mut(
        &mut self,
    ) -> core::result::Result<&mut ResourceOwner<Resource>, CompletionError<Resource::Error>> {
        self.owner
            .as_deref_mut()
            .ok_or(CompletionError::MissingResource)
    }

    fn resource(
        &mut self,
    ) -> core::result::Result<&mut Resource, CompletionError<Resource::Error>> {
        let owner = self.owner_mut()?;
        match (owner.state, owner.resource.as_mut()) {
            (ResourceState::InFlight, Some(resource)) => Ok(resource),
            _ => Err(CompletionError::MissingResource),
        }
    }

    /// Marks the bundle in flight before its first device submission.
    fn mark_submitted(&mut self) {
        self.submitted = true;
    }

    /// Synchronizes submitted work, then runs the checked prepublication step.
    fn complete<Value>(
        self,
        commit: impl FnOnce(&mut Resource) -> core::result::Result<Value, Resource::Error>,
    ) -> core::result::Result<Value, CompletionError<Resource::Error>> {
        let mut completion = self.complete_prepublication()?;
        let value = match commit(completion.resource()) {
            Ok(value) => value,
            Err(source) => {
                return Err(CompletionError::Commit { source });
            }
        };
        Ok(completion.publish(|_| value))
    }

    /// Prove completion before a transaction's fallible shared preparation.
    ///
    /// Dropping the returned phase never synchronizes a second time. It keeps
    /// the complete resource as known-idle poisoned custody until its caller
    /// performs the one infallible publication tail.
    fn complete_prepublication(
        mut self,
    ) -> core::result::Result<PostCompletion<'owner, Resource>, CompletionError<Resource::Error>>
    {
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
        let owner = self.owner.take().ok_or(CompletionError::MissingResource)?;
        let (state, resource) = (&mut owner.state, &mut owner.resource);
        let resource = resource.as_mut().ok_or(CompletionError::MissingResource)?;
        self.finished = true;
        Ok(PostCompletion {
            state,
            resource,
            finished: false,
        })
    }

    fn poison_uncertain(&mut self) {
        let Ok(owner) = self.owner_mut() else {
            self.finished = true;
            return;
        };
        if owner.state != ResourceState::InFlight || owner.resource.is_none() {
            self.finished = true;
            return;
        }
        owner.state = ResourceState::PoisonedUncertain;
        self.finished = true;
    }

    fn poison_known_idle(&mut self) {
        let Ok(owner) = self.owner_mut() else {
            self.finished = true;
            return;
        };
        if owner.state != ResourceState::InFlight || owner.resource.is_none() {
            self.finished = true;
            return;
        }
        owner.state = ResourceState::PoisonedIdle;
        self.finished = true;
    }

    fn finish_after_submission(&mut self) {
        let Ok(owner) = self.owner_mut() else {
            self.finished = true;
            return;
        };
        if owner.state != ResourceState::InFlight {
            self.finished = true;
            return;
        }
        let Some(resource) = owner.resource.as_mut() else {
            self.finished = true;
            return;
        };
        owner.state = match resource.synchronize() {
            Ok(()) => ResourceState::PoisonedIdle,
            Err(_) => ResourceState::PoisonedUncertain,
        };
        self.finished = true;
    }
}

struct PostCompletion<'owner, Resource: CompletionResource> {
    state: &'owner mut ResourceState,
    resource: &'owner mut Resource,
    finished: bool,
}

impl<Resource: CompletionResource> PostCompletion<'_, Resource> {
    fn resource(&mut self) -> &mut Resource {
        self.resource
    }

    /// Publish an already prepared resource without allocation, validation, or
    /// callbacks after the caller's final infallible commit.
    fn publish<Value>(mut self, publish: impl FnOnce(&mut Resource) -> Value) -> Value {
        let value = publish(self.resource);
        *self.state = ResourceState::Ready;
        self.finished = true;
        value
    }
}

impl<Resource: CompletionResource> Drop for PostCompletion<'_, Resource> {
    fn drop(&mut self) {
        if !self.finished {
            *self.state = ResourceState::PoisonedIdle;
            self.finished = true;
        }
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
        let Some(owner) = self.owner.as_deref_mut() else {
            return;
        };
        if owner.state != ResourceState::InFlight || owner.resource.is_none() {
            return;
        }
        owner.state = ResourceState::Ready;
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
        assert_eq!(owner.state(), ResourceState::Ready);
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
        assert_eq!(owner.state(), ResourceState::Ready);
        assert_eq!(synchronizations.get(), 1);
        assert_eq!(validations.get(), 1);
        assert_eq!(publications.get(), 1);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn post_completion_drop_retains_known_idle_without_a_second_sync()
    -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            validations,
            drops,
            publications,
        } = owner([Ok(())], [Ok(())]);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        let completion = guard
            .complete_prepublication()
            .map_err(|_| BeginError::MissingResource)?;
        drop(completion);
        assert_eq!(owner.state(), ResourceState::PoisonedIdle);
        assert_eq!(synchronizations.get(), 1);
        assert_eq!(validations.get(), 1);
        assert_eq!(publications.get(), 0);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }

    #[test]
    fn post_completion_publish_is_infallible_and_restores_ready()
    -> core::result::Result<(), BeginError> {
        let OwnerFixture {
            mut owner,
            synchronizations,
            validations,
            drops,
            publications,
        } = owner([Ok(())], [Ok(())]);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        let completion = guard
            .complete_prepublication()
            .map_err(|_| BeginError::MissingResource)?;
        completion.publish(|resource| resource.publish());
        assert_eq!(owner.state(), ResourceState::Ready);
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
        assert_eq!(owner.state(), ResourceState::Ready);
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
        assert!(matches!(owner.state(), ResourceState::PoisonedIdle));
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
        assert!(matches!(owner.state(), ResourceState::PoisonedIdle));
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
        assert!(matches!(owner.state(), ResourceState::PoisonedUncertain));
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
        assert!(matches!(owner.state(), ResourceState::PoisonedUncertain));
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
        assert!(matches!(owner.state(), ResourceState::PoisonedIdle));
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
        } = owner([Ok(())], [Ok(())]);
        let mut guard = owner.begin()?;
        guard.mark_submitted();
        // NOTE: completion already proved idle; unwind poisons the retained
        // bundle without issuing a second synchronization.
        let unwind = catch_unwind(AssertUnwindSafe(|| {
            drop(guard.complete(|_| -> core::result::Result<(), TestError> {
                panic_any(ControlledUnwind)
            }));
        }));
        assert!(unwind.is_err());
        assert!(matches!(owner.state(), ResourceState::PoisonedIdle));
        assert_eq!(publications.get(), 0);
        assert_eq!(synchronizations.get(), 1);
        drop(owner);
        assert_eq!(drops.get(), 1);
        Ok(())
    }
}
