use std::fmt;
use std::sync::Arc;

type Guard = dyn Fn(&mut dyn FnMut()) + Send + Sync;

/// Wraps calls that create threads inside third-party code, which the
/// service cannot reach once they run, so the embedding process can
/// keep that creation away from its own thread placement.
#[derive(Clone, Default)]
pub struct CreationGuard(Option<Arc<Guard>>);

impl CreationGuard {
    pub fn new(guard: impl Fn(&mut dyn FnMut()) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(guard)))
    }

    pub(crate) fn run<R>(&self, create: impl FnOnce() -> R) -> R {
        let Some(guard) = &self.0 else {
            return create();
        };
        let mut create = Some(create);
        let mut created = None;
        guard(&mut || created = create.take().map(|create| create()));
        created.expect("the creation guard runs its work exactly once")
    }
}

impl fmt::Debug for CreationGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CreationGuard")
            .field(&self.0.is_some())
            .finish()
    }
}

/// Puts the calling thread back on every processor the process may use,
/// undoing any placement it was given when it was created.
pub(crate) fn release_affinity() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess,
            GetCurrentThread,
            GetProcessAffinityMask,
            SetThreadAffinityMask,
        };

        let mut process = 0;
        let mut system = 0;
        if GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) != 0 {
            SetThreadAffinityMask(GetCurrentThread(), process);
        }
    }
}
