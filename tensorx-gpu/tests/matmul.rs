use tensorx_gpu::{context::GpuContext, tensor_gpu::TensorGpu};

#[test]
fn test_matmul_gpu() {
    let ctx = GpuContext::new();
    
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];

    let a = TensorGpu::from_data(&ctx, &a_data, vec![2, 2]);
    let b = TensorGpu::from_data(&ctx, &b_data, vec![2, 2]);
    let c = TensorGpu::matmul(&ctx, &a, &b);
    let c_host = c.to_vec(&ctx);

    let expected = vec![
        1.0 * 5.0 + 2.0 * 7.0,
        1.0 * 6.0 + 2.0 * 8.0,
        3.0 * 5.0 + 4.0 * 7.0,
        3.0 * 6.0 + 4.0 * 8.0,
    ];


    assert_eq!(expected, c_host);

    // for (i, (x, y)) in c_host.iter().zip(expected.iter()).enumerate() {
    //     assert!((x - y).abs() < 1e-4, "Mismatch at {}: {} != {}", i, x, y);
    // }
}
