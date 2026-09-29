use std::sync::Mutex;

use av_denoise_core::{FrameLayout, Planes};

pub(crate) struct PlanesPool {
    layout: FrameLayout,
    free: Mutex<Vec<Planes>>,
}

impl PlanesPool {
    pub(crate) fn new(layout: FrameLayout) -> Self {
        Self {
            layout,
            free: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn take(&self) -> Planes {
        self.free
            .lock()
            .expect("plane pool lock is never poisoned")
            .pop()
            .unwrap_or_default()
    }

    pub(crate) fn recycle(&self, planes: Planes) {
        if planes.y.len() != self.layout.luma_bytes()
            || planes.u.len() != self.layout.chroma_bytes()
            || planes.v.len() != self.layout.chroma_bytes()
        {
            return;
        }
        self.free
            .lock()
            .expect("plane pool lock is never poisoned")
            .push(planes);
    }
}

pub(crate) struct SharedPools {
    pub(crate) staged: PlanesPool,
    pub(crate) emitted: PlanesPool,
}

impl SharedPools {
    pub(crate) fn new(staged: FrameLayout, emitted: FrameLayout) -> Self {
        Self {
            staged: PlanesPool::new(staged),
            emitted: PlanesPool::new(emitted),
        }
    }
}
