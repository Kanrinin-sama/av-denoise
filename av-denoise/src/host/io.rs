use av_denoise_core::runtime::{ComputeClient, Runtime};
use std::future::Future;
use std::pin::Pin;

use cubecl::bytes::Bytes;
use cubecl::server::{Handle, ServerError};

/// A readback in flight, resolving to one buffer per handle read.
pub(crate) type ReadFuture = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send>>;

/// Uploads and reads back planes on whichever runtime an engine was built on.
pub(crate) trait PlaneIo: Send {
    fn upload(&mut self, planes: &[&[u8]]) -> Vec<Handle>;

    fn allocate(&self, bytes: usize) -> Handle;

    /// Starts reading `handles` back to the host.
    ///
    /// Nothing is read until the returned future is first polled.
    fn read(&self, handles: Vec<Handle>) -> ReadFuture;
}

pub(crate) struct ClientIo<R: Runtime> {
    client: ComputeClient<R>,
    input: Option<av_denoise_core::upload::Uploader<R>>,
}

impl<R: Runtime> ClientIo<R> {
    pub(crate) fn new(client: ComputeClient<R>) -> Self {
        Self { client, input: None }
    }
}

impl<R: Runtime> PlaneIo for ClientIo<R> {
    fn upload(&mut self, planes: &[&[u8]]) -> Vec<Handle> {
        let input = self.input.get_or_insert_with(|| {
            let sizes = planes.iter().map(|plane| plane.len()).collect::<Vec<_>>();
            av_denoise_core::upload::Uploader::new(&self.client, &sizes)
        });
        let contents = planes.iter().map(|plane| vec![*plane]).collect::<Vec<_>>();
        let contents = contents.iter().map(Vec::as_slice).collect::<Vec<_>>();
        input.upload(&contents);
        (0..planes.len())
            .map(|index| input.handle(index).clone())
            .collect()
    }

    fn allocate(&self, bytes: usize) -> Handle {
        self.client.empty(bytes)
    }

    fn read(&self, handles: Vec<Handle>) -> ReadFuture {
        // The future owns its own client, so it stays valid after the denoiser that started it is dropped.
        let client = self.client.clone();
        let future = async move { client.read_async(handles).await };

        Box::pin(future)
    }
}
