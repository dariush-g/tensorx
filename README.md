# tensr

**tensr** is an experimental, modular tensor computation library written in Rust, with early support for GPU acceleration via `wgpu`. It’s designed as a learning project and a prototype—not a production-ready library.

## Features

- Modular crate structure (`tensr-core`, `tensr-gpu`, `tensr`)
- Basic tensor operations
- CPU and GPU support for operations
- Built for learning and experimentation

## Example

```rust
use tensr::Tensor;

fn main() {
    let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    let b = Tensor::from([[5.0, 6.0], [7.0, 8.0]]);
    let c = a.matmul(&b);

    println!("{:?}", c);
}
