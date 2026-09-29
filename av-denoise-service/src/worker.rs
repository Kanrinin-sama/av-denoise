use std::collections::VecDeque;
use std::thread;

use av_denoise_core::{FrameLayout, PlanarDenoiser, PlaneOptions, Planes, WarmUp, push_needs_retry};
use crossbeam_channel::{Receiver, Sender};

use crate::budget::Permit;
use crate::dispatch::SceneJob;
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
/// denoised frames on `output`.
pub(crate) fn spawn_workers(
    opts: &PlaneOptions,
    layout: FrameLayout,
    denoisers: Vec<Option<Resident>>,
    output: Sender<OutputMsg>,
) -> (Sender<SceneJob>, Vec<WorkerJoin>) {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(denoisers.len());

    for (worker_id, denoiser) in denoisers.into_iter().enumerate() {
        let opts = opts.clone();
        let out_tx = output.clone();
        let job_rx = job_rx.clone();

        worker_handles.push(thread::spawn(move || {
            run_worker(worker_id, opts, layout, denoiser, job_rx, out_tx)
        }));
    }

    (job_tx, worker_handles)
}

fn run_worker(
    worker_id: usize,
    opts: PlaneOptions,
    layout: FrameLayout,
    mut wd: Option<Resident>,
    jobs: Receiver<SceneJob>,
    tx: Sender<OutputMsg>,
) -> Result<Option<Resident>, anyhow::Error> {
    while let Ok(job) = jobs.recv() {
        // Built on the first claimed scene unless the slot built it ahead.
        if wd.is_none() {
            wd = Some(create_denoiser(&opts, layout)?);
        }

        let (denoiser, warm_up) = wd.as_mut().expect("denoiser exists after the check above");

        tracing::debug!(worker_id, scene_idx = job.scene_idx, "worker started scene");

        let mut pending = Pending::new();

        // Nothing is received straight after the push.
        // `push_with_drain` handles backpressure through QueueFull
        // when the 2-deep pending pipeline fills, and `flush_worker`
        // drains the tail below. Receiving after every push would clamp
        // the pipeline back to depth 1 and put the GPU readback in the
        // critical path of the next push.
        for frame in job.frames {
            pending.push_back((frame.global_idx, frame.permit));
            push_with_drain(denoiser, warm_up, &mut pending, &frame.planes, &tx)?;
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
    tx: &Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    if push_needs_retry(denoiser.push(planes))? {
        if let Some(out) = denoiser.recv()? {
            let (global_idx, permit) = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            tx.send(OutputMsg {
                global_idx,
                planes: out,
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
