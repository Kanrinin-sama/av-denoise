use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use cubecl::client::Client;
pub use cubecl_environment::stream::StreamId;
pub use cubecl_runtime::runtime::Runtime;

pub struct ComputeClient<R: Runtime> {
    client: Client,
    runtime: PhantomData<R>,
}

impl<R: Runtime> ComputeClient<R> {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            runtime: PhantomData,
        }
    }
}

impl<R: Runtime> Clone for ComputeClient<R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            runtime: PhantomData,
        }
    }
}

impl<R: Runtime> Deref for ComputeClient<R> {
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl<R: Runtime> DerefMut for ComputeClient<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.client
    }
}
