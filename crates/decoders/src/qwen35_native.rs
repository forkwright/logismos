//! Owned native full-attention resource lifecycle.

#[cfg(feature = "gpu")]
mod plan;
#[cfg(feature = "gpu")]
mod weights;

/// One resource bundle whose submitted work can be synchronized.
///
/// This private seam lets the native session exercise the same ownership and
/// drop path under fault injection without manufacturing a second state model.
trait CompletionResource {
    type Error;

    fn synchronize(&mut self) -> core::result::Result<(), Self::Error>;
}

enum ResourceState<Resource> {
    Ready(Resource),
    InFlight(Resource),
    PoisonedIdle(Resource),
    PoisonedUncertain(Resource),
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

    fn begin(&mut self) -> Option<InFlight<'_, Resource>> {
        let state = self.state.take()?;
        self.state = Some(match state {
            ResourceState::Ready(resource) => ResourceState::InFlight(resource),
            state => state,
        });
        matches!(self.state, Some(ResourceState::InFlight(_))).then_some(InFlight {
            owner: self,
            submitted: false,
            complete: false,
        })
    }

    fn state(&self) -> Option<&ResourceState<Resource>> {
        self.state.as_ref()
    }
}

impl<Resource: CompletionResource> Drop for ResourceOwner<Resource> {
    fn drop(&mut self) {
        let Some(ResourceState::PoisonedUncertain(mut resource)) = self.state.take() else {
            return;
        };
        if resource.synchronize().is_err() {
            core::mem::forget(resource);
        }
    }
}

struct InFlight<'owner, Resource: CompletionResource> {
    owner: &'owner mut ResourceOwner<Resource>,
    submitted: bool,
    complete: bool,
}

impl<Resource: CompletionResource> InFlight<'_, Resource> {
    fn resource(&mut self) -> Option<&mut Resource> {
        match self.owner.state.as_mut() {
            Some(ResourceState::InFlight(resource)) => Some(resource),
            Some(ResourceState::Ready(_))
            | Some(ResourceState::PoisonedIdle(_))
            | Some(ResourceState::PoisonedUncertain(_))
            | None => None,
        }
    }

    fn mark_submitted(&mut self) {
        self.submitted = true;
    }

    fn complete(mut self) -> core::result::Result<(), Resource::Error> {
        let Some(resource) = self.resource() else {
            self.complete = true;
            return Ok(());
        };
        if let Err(error) = resource.synchronize() {
            self.poison();
            return Err(error);
        }
        self.complete = true;
        Ok(())
    }

    fn poison(&mut self) {
        let Some(ResourceState::InFlight(mut resource)) = self.owner.state.take() else {
            return;
        };
        self.owner.state = Some(match resource.synchronize() {
            Ok(()) => ResourceState::PoisonedIdle(resource),
            Err(_) => ResourceState::PoisonedUncertain(resource),
        });
        self.complete = true;
    }
}

impl<Resource: CompletionResource> Drop for InFlight<'_, Resource> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        if self.submitted {
            self.poison();
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
    use std::rc::Rc;

    use super::{CompletionResource, ResourceOwner, ResourceState};

    struct FaultResource {
        synchronizations: Rc<Cell<usize>>,
        fail: bool,
    }

    impl CompletionResource for FaultResource {
        type Error = ();

        fn synchronize(&mut self) -> core::result::Result<(), Self::Error> {
            self.synchronizations
                .set(self.synchronizations.get().saturating_add(1));
            if self.fail { Err(()) } else { Ok(()) }
        }
    }

    #[test]
    fn submitted_guard_retains_uncertain_resource_until_owner_drop()
    -> core::result::Result<(), String> {
        let synchronizations = Rc::new(Cell::new(0));
        let mut owner = ResourceOwner::new(FaultResource {
            synchronizations: Rc::clone(&synchronizations),
            fail: true,
        });
        let mut guard = owner
            .begin()
            .ok_or_else(|| "ready owner must begin one guarded step".to_string())?;
        guard.mark_submitted();
        drop(guard);
        assert!(
            matches!(owner.state(), Some(ResourceState::PoisonedUncertain(_))),
            "a failed submitted synchronization must retain the actual resource bundle"
        );
        assert_eq!(
            synchronizations.get(),
            1,
            "the production in-flight guard must synchronize before declaring uncertainty"
        );
        Ok(())
    }
}
