use std::marker::PhantomData;

use tensr_core::tensor::compute_strides;
use wgpu::{Buffer, BufferUsages, util::DeviceExt};

use crate::{context::GpuContext, matmul::matmul_gpu};

pub struct TensorGpu<T> {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub len: usize,
    pub buffer: GpuBuffer<T>,
}

impl<T> TensorGpu<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl<T: bytemuck::Pod> TensorGpu<T> {
    pub fn to_vec(&self, ctx: &GpuContext) -> Vec<T> {
        self.buffer.to_vec(ctx)
    }
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    pub fn to_vec(&self, ctx: &GpuContext) -> Vec<T> {
        let buffer_size = (self.len * std::mem::size_of::<T>()) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("StagingBuffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("to_vec encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, buffer_size);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        result
    }
}

impl TensorGpu<f32> {
    pub fn from_data(ctx: &GpuContext, data: &[f32], shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        assert_eq!(data.len(), len);
        let strides = compute_strides(&shape); // from tensr-core
        let buffer = GpuBuffer::from_data(ctx, data);
        Self {
            shape,
            strides,
            len,
            buffer,
        }
    }

    pub fn matmul(ctx: &GpuContext, a: &Self, b: &Self) -> Self {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        let (m, k1) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        assert_eq!(k1, k2);

        let result = matmul_gpu(ctx, &a.buffer, &b.buffer, m, k1, n);

        let shape = vec![m, n];
        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            len: m * n,
            buffer: result,
        }
    }
}

pub struct GpuBuffer<T> {
    pub buffer: Buffer,
    pub len: usize,
    pub _marker: PhantomData<T>,
}

impl<T> GpuBuffer<T>
where
    T: bytemuck::Pod,
{
    pub fn from_data(context: &GpuContext, data: &[T]) -> Self {
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuBuffer"),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

        Self {
            buffer,
            len: data.len(),
            _marker: PhantomData,
        }
    }
}
