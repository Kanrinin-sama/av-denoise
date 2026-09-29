use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use av_denoise_core::{FrameLayout, PlanarDenoiser, PlaneOptions, Planes, WarmUp, push_needs_retry};
use crossbeam_channel::{Receiver, Sender};

use crate::CreationGuard;
use crate::budget::Permit;
use crate::cancel::Cancel;
use crate::dispatch::{SceneJob, StagedFrame};
use crate::pool::SharedPools;
use crate::threads::release_affinity;
use crate::warm::{create_denoiser, finish_warm_up};

/// One denoised frame on its way to the window, still holding its permit.
pub(crate) struct OutputMsg {
    pub(crate) global_idx: u64,
    pub(crate) planes: Planes,
    pub(crate) _permit: Permit,
}

/// Indices of pushed-but-not-yet-emitted frames, in push order, with the
/// permits they hold.
type Pending = VecDeque<(u64, Permit)>;

/// A built denoiser with the cold-cache queue place it holds until its
/// first output frame proves the kernels are compiled and cached.
pub(crate) type Resident = (PlanarDenoiser, Option<WarmUp>);

pub(crate) type WorkerJoin = thread::JoinHandle<Result<Option<Resident>, anyhow::Error>>;

/// Spawns one worker thread per denoiser slot over one shared scene queue.
///
/// The queue is a rendezvous, so a scene is only offered when a worker is
/// free and at most `workers` scenes are ever in flight.
///
/// Returns the queue's sender and their join handles. Workers emit
/// denoised frames on `output`, stop taking frames once `cancel` is set,
/// and set `abort` when one of them fails.
pub(crate) struct WorkerConfig<'a> {
    pub(crate) opts: &'a PlaneOptions,
    pub(crate) layout: FrameLayout,
    pub(crate) output: Sender<OutputMsg>,
    pub(crate) cancel: &'a Cancel,
    pub(crate) abort: &'a Arc<AtomicBool>,
    pub(crate) guard: &'a CreationGuard,
    pub(crate) pools: &'a Arc<SharedPools>,
}

pub(crate) fn spawn_workers(
    config: &WorkerConfig<'_>,
    denoisers: Vec<Option<Resident>>,
) -> (Sender<SceneJob>, Vec<WorkerJoin>) {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(denoisers.len());

    for (worker_id, denoiser) in denoisers.into_iter().enumerate() {
        let opts = config.opts.clone();
        let layout = config.layout;
        let out_tx = config.output.clone();
        let job_rx = job_rx.clone();
        let cancel = config.cancel.clone();
        let abort = Arc::clone(config.abort);
        let guard = config.guard.clone();
        let pools = Arc::clone(config.pools);

        worker_handles.push(thread::spawn(move || {
            release_affinity();
            let result = run_worker(
                worker_id,
                || guard.run(|| create_denoiser(&opts, layout)),
                denoiser,
                job_rx,
                out_tx,
                &cancel,
                &pools,
            );
            if result.is_err() {
                abort.store(true, Ordering::Release);
            }
            result
        }));
    }

    (job_tx, worker_handles)
}

fn run_worker(
    worker_id: usize,
    create: impl Fn() -> Result<Resident, anyhow::Error>,
    mut wd: Option<Resident>,
    jobs: Receiver<SceneJob>,
    tx: Sender<OutputMsg>,
    cancel: &Cancel,
    pools: &SharedPools,
) -> Result<Option<Resident>, anyhow::Error> {
    let mut emitted = pools.emitted.take();
    while let Ok(job) = jobs.recv() {
        if cancel.is_set() {
            continue;
        }

        // Built on the first claimed scene unless the slot built it ahead.
        if wd.is_none() {
            wd = Some(create()?);
        }

        let (denoiser, warm_up) = wd.as_mut().expect("denoiser exists after the check above");

        tracing::debug!(worker_id, scene_idx = job.scene_idx, "worker started scene");

        let mut pending = Pending::new();
        let mut pushed = false;

        // Nothing is received straight after the push.
        // `push_with_drain` handles backpressure through QueueFull
        // when the 2-deep pending pipeline fills, and `flush_worker`
        // drains the tail below. Receiving after every push would clamp
        // the pipeline back to depth 1 and put the GPU readback in the
        // critical path of the next push.
        for frame in job.frames {
            if cancel.is_set() {
                break;
            }
            let StagedFrame {
                global_idx,
                planes: staged,
                permit,
            } = frame;
            pending.push_back((global_idx, permit));
            push_with_drain(
                denoiser,
                warm_up,
                &mut pending,
                &staged,
                pools,
                &tx,
                &mut emitted,
            )?;
            pools.staged.recycle(staged);
            pushed = true;
        }

        if !pushed {
            continue;
        }

        // Reuse the PlanarDenoiser across scenes. Flushing here ensures
        // no temporal window spans two of them.
        flush_worker(denoiser, warm_up, &mut pending, &tx)?;
    }

    Ok(wd)
}

/// Push one frame, draining any pending output first if the queue is full.
fn push_with_drain(
    denoiser: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut Pending,
    planes: &Planes,
    pools: &SharedPools,
    tx: &Sender<OutputMsg>,
    emitted: &mut Planes,
) -> Result<(), anyhow::Error> {
    if push_needs_retry(denoiser.push(planes))? {
        if denoiser.recv_into(emitted)? {
            let (global_idx, permit) = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            tx.send(OutputMsg {
                global_idx,
                planes: std::mem::replace(emitted, pools.emitted.take()),
                _permit: permit,
            })
            .map_err(|_| anyhow::anyhow!("coordinator disconnected"))?;
            finish_warm_up(warm_up);
        }

        denoiser.push(planes)?;
    }

    Ok(())
}

fn flush_worker(
    wd: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut Pending,
    tx: &Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut disconnected = false;

    wd.flush(|out| {
        if disconnected {
            return;
        }

        if let Some((global_idx, permit)) = pending.pop_front() {
            let msg = OutputMsg {
                global_idx,
                planes: out,
                _permit: permit,
            };
            let did_send = tx.send(msg).is_ok();
            if did_send {
                finish_warm_up(warm_up);
            } else {
                disconnected = true;
            }
        } else {
            tracing::warn!("worker emitted flushed frame with no pending global index");
        }
    })?;

    if disconnected {
        anyhow::bail!("coordinator disconnected while flushing worker output");
    }

    Ok(())
}
