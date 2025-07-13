use tensorx_core::tensor::Tensor;

#[test]
fn test_matmul() {
    let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    let b = Tensor::from([[5.0, 6.0], [7.0, 8.0]]);
    let c = a.matmul(&b);

    //
    //  [ 1.0 , 2.0 ]
    //  [ 3.0 , 4.0 ]
    //
    //  [ 5.0 , 6.0 ]
    //  [ 7.0 , 8.0 ]
    //

    assert_eq!(c.unwrap().get_data(), vec![19., 22., 43., 50.]);
}
