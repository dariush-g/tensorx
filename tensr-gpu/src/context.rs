use std::sync::Arc;

use wgpu::{Device, Queue};

pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl GpuContext {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }
}
