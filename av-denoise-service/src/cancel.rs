use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crossbeam_channel::{Receiver, Sender};

#[derive(Clone, Debug)]
pub struct Cancellation(Arc<Signal>);

#[derive(Debug)]
struct Signal {
    flag: Arc<AtomicBool>,
    receiver: Receiver<()>,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    sender: Option<Sender<()>>,
    listeners: Vec<Weak<Signal>>,
}

impl Cancellation {
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam_channel::bounded(0);
        Self(Arc::new(Signal {
            flag: Arc::new(AtomicBool::new(false)),
            receiver,
            state: Mutex::new(State {
                sender: Some(sender),
                listeners: Vec::new(),
            }),
        }))
    }

    pub fn cancel(&self) {
        self.0.flag.store(true, Ordering::Release);
        let listeners = {
            let mut state = self.0.state.lock().unwrap_or_else(|error| error.into_inner());
            let Some(sender) = state.sender.take() else {
                return;
            };
            drop(sender);
            std::mem::take(&mut state.listeners)
        };
        for listener in listeners {
            if let Some(signal) = listener.upgrade() {
                Self(signal).cancel();
            }
        }
    }

    pub fn is_set(&self) -> bool {
        self.0.flag.load(Ordering::Acquire)
    }

    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0.flag)
    }

    fn listen(&self, listener: &Self) {
        let mut state = self.0.state.lock().unwrap_or_else(|error| error.into_inner());
        if self.is_set() {
            drop(state);
            listener.cancel();
        } else {
            state.listeners.retain(|entry| entry.strong_count() > 0);
            state.listeners.push(Arc::downgrade(&listener.0));
        }
    }
}

impl Default for Cancellation {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub(crate) struct Cancel {
    signal: Cancellation,
    _parents: Vec<Cancellation>,
}

impl Cancel {
    pub(crate) fn new(flags: Vec<Cancellation>) -> Self {
        let signal = Cancellation::new();
        for flag in &flags {
            flag.listen(&signal);
        }
        Self {
            signal,
            _parents: flags,
        }
    }

    pub(crate) fn is_set(&self) -> bool {
        self.signal.is_set()
    }

    pub(crate) fn receiver(&self) -> &Receiver<()> {
        &self.signal.0.receiver
    }
}
