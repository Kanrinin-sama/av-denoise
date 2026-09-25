//! Lets one process fill a cold kernel cache while the others wait.
//!
//! CubeCL compiles a kernel the first time it is dispatched, and writes
//! the result to the cache [`install_compilation_cache`] points it at.
//! The table of contents that cache reads is a snapshot taken when the
//! GPU client is built, so a process that starts while a second process
//! is still compiling sees an empty table and shares nothing with it.
//! Every process in that group pays the full compilation cost and
//! appends its own copy of the same kernels.
//!
//! [`WarmUp`] closes that window with a lock file next to the cache.
//! The first process in holds it while it compiles, the rest block, and
//! by the time they build their own client the cache has everything they
//! need. Once a run finishes it leaves a stamp file behind, and later
//! processes see the stamp and skip the lock entirely, so the cost is
//! paid once rather than on every chunk.
//!
//! [`install_compilation_cache`]: crate::install_compilation_cache

use std::collections::HashSet;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use cubecl::hash::StableHasher;

use crate::cache::compilation_cache_dir;
use crate::frame::{FrameLayout, PlaneOptions};

/// How long a process waits for the one ahead of it before giving up and
/// compiling for itself.
///
/// Compiling this crate's kernels takes about ten seconds on a quiet
/// machine, and a machine running an encode is not quiet, so the limit
/// is generous. Waiting longer than this is worse than duplicating the
/// work, because the encoder above has nothing to do until a frame
/// arrives.
const WAIT_LIMIT: Duration = Duration::from_secs(180);

/// How often a waiting process retries the lock.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The keys this process already holds a place for.
static CLAIMED_KEYS: LazyLock<Mutex<HashSet<u128>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// Records that this process is taking the place for `key`, and reports
/// whether it was free to take.
fn claim_key(key: u128) -> bool {
    CLAIMED_KEYS
        .lock()
        .expect("warm-up key mutex poisoned")
        .insert(key)
}

/// Gives `key` back, so a later filter in this process can queue for it
/// again.
fn release_key(key: u128) {
    CLAIMED_KEYS
        .lock()
        .expect("warm-up key mutex poisoned")
        .remove(&key);
}

/// Identifies the set of kernels a denoiser compiles.
///
/// Radii, channel mode, depth and algorithm are all baked into the
/// kernels at compile time, so a change to any of them produces a
/// different set and a warm cache for one set says nothing about
/// another. Processes that would compile different kernels have no
/// reason to wait for each other, and a stamp left by one must not
/// convince the other that its own kernels are cached.
///
/// The key is taken from the `Debug` rendering of both inputs rather
/// than from a hand-written list of the fields that reach a kernel.
/// A hand-written list silently stops covering a field the day someone
/// adds one, and the failure that follows is a process trusting a stamp
/// for kernels it has never compiled. Reading everything keeps the key
/// correct without maintenance.
///
/// Including fields that only ever reach the GPU as runtime arguments,
/// such as strength, makes the key finer than it strictly needs to be.
/// Two runs that differ only in strength warm up separately instead of
/// sharing. That costs one extra warm-up and is the safe direction to
/// err in.
///
/// This crate's version goes into the key as well, because a release
/// that changes a kernel changes what CubeCL caches under it, and
/// CubeCL files its own cache under the CubeCL version on top of that.
/// Without the version a stamp written before an upgrade would tell
/// every process after it that a cache emptied by that upgrade is warm,
/// and the whole first wave would compile at once again with nothing
/// said about it. Rebuilding a kernel without changing the version is
/// the one case this misses, and deleting the cache directory clears it.
pub fn kernel_key(options: &PlaneOptions, layout: FrameLayout) -> u128 {
    StableHasher::hash_one(&format!("{}|{options:?}|{layout:?}", env!("CARGO_PKG_VERSION")))
}

/// A held place in the queue to fill a cold cache.
///
/// Obtained from [`WarmUp::begin`] and given up with [`WarmUp::finish`]
/// once the kernels are compiled. Dropping one without calling `finish`
/// releases the lock without leaving a stamp, so a run that failed part
/// way through does not convince the next process that the cache is
/// warm.
#[derive(Debug)]
pub struct WarmUp {
    lock: File,
    stamp: PathBuf,
    key: u128,
}

impl WarmUp {
    /// Takes a place in the queue for the kernels `key` identifies.
    ///
    /// Returns `Some` while holding the lock, and the caller compiles
    /// under it. Returns `None` when there is nothing to wait for, which
    /// covers a cache that is already warm for these kernels, a cache
    /// that is turned off, and a lock that could not be taken in
    /// [`WAIT_LIMIT`]. In every one of those the caller carries on and
    /// compiles as it always did.
    ///
    /// Blocks for as long as the process ahead takes to compile, so it
    /// belongs on the path that builds a denoiser rather than on the
    /// path that renders a frame.
    pub fn begin(key: u128) -> Option<Self> {
        Self::begin_in(compilation_cache_dir()?, key, WAIT_LIMIT)
    }

    /// [`WarmUp::begin`] against an explicit directory and wait limit.
    ///
    /// Split out so tests can drive the queue without the process-wide
    /// cache directory, and without waiting the full [`WAIT_LIMIT`] to
    /// see what a contended lock does.
    fn begin_in(dir: &Path, key: u128, wait_limit: Duration) -> Option<Self> {
        // A file lock is held by the process rather than by the handle
        // that took it, so a second filter in this process asking for
        // the same kernels would wait out `wait_limit` on a lock its own
        // process already holds. One script can easily build two
        // filters, so take the place at most once per key per process
        // and let the second caller carry on.
        if !claim_key(key) {
            return None;
        }

        let held = Self::acquire(dir, key, wait_limit);

        if held.is_none() {
            release_key(key);
        }

        held
    }

    /// [`WarmUp::begin_in`] without the in-process bookkeeping, which its
    /// caller takes care of on both the success and the failure path.
    fn acquire(dir: &Path, key: u128, wait_limit: Duration) -> Option<Self> {
        let stamp = dir.join(format!("warm-{key:032x}.stamp"));

        if stamp.exists() {
            return None;
        }

        let lock = open_lock_file(&dir.join(format!("warm-{key:032x}.lock")))?;

        if !wait_for_lock(&lock, wait_limit) {
            return None;
        }

        // Whoever held the lock has finished compiling by now, so the
        // stamp answers differently than it did above. Checking it again
        // is what turns the queue into a single warm-up rather than one
        // per waiting process.
        if stamp.exists() {
            let _ = lock.unlock();
            return None;
        }

        tracing::debug!(?stamp, "compiling kernels for a cold cache");
        Some(Self { lock, stamp, key })
    }

    /// Records that the kernels are compiled and lets the next process
    /// through.
    pub fn finish(self) {
        if let Err(err) = std::fs::write(&self.stamp, b"") {
            // The next process reads a missing stamp as a cold cache and
            // compiles again, which is slower but still correct.
            tracing::debug!(stamp = ?self.stamp, %err, "cannot write the kernel warm-up stamp");
        }
    }
}

impl Drop for WarmUp {
    fn drop(&mut self) {
        let _ = self.lock.unlock();
        release_key(self.key);
    }
}

/// Opens the lock file, creating it when this is the first process to
/// ask for these kernels.
///
/// A directory that cannot be written is reported and then ignored, the
/// same way an uncreatable cache directory is. Denoising works without
/// the queue, it just compiles more than once.
fn open_lock_file(path: &Path) -> Option<File> {
    match OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
    {
        Ok(file) => Some(file),
        Err(err) => {
            tracing::debug!(?path, %err, "cannot open the kernel warm-up lock, compiling unqueued");
            None
        },
    }
}

/// Blocks until the lock is held, giving up after `wait_limit`.
///
/// Returns whether the lock is held. The lock is a real advisory file
/// lock rather than a file whose presence means "taken", so the
/// operating system releases it when Av1an kills a worker mid-compile
/// and the next process in line wakes up straight away.
fn wait_for_lock(lock: &File, wait_limit: Duration) -> bool {
    let start = Instant::now();

    loop {
        match lock.try_lock() {
            Ok(()) => return true,
            Err(TryLockError::WouldBlock) => {},
            Err(TryLockError::Error(err)) => {
                tracing::debug!(%err, "cannot take the kernel warm-up lock, compiling unqueued");
                return false;
            },
        }

        if start.elapsed() >= wait_limit {
            tracing::warn!(
                "waited {:?} for another process to compile kernels, compiling for ourselves",
                wait_limit,
            );
            return false;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}
