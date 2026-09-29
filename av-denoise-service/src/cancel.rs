use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The flags that stop a window, any one of which is enough.
#[derive(Clone)]
pub(crate) struct Cancel(Vec<Arc<AtomicBool>>);

impl Cancel {
    pub(crate) fn new(flags: Vec<Arc<AtomicBool>>) -> Self {
        Self(flags)
    }

    pub(crate) fn is_set(&self) -> bool {
        self.0.iter().any(|flag| flag.load(Ordering::Acquire))
    }
}
