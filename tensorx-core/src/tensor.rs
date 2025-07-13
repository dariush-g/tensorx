// use serde::Serialize;
// use serde::de::DeserializeOwned;

use crate::error::{Result, TensorError};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}



impl<T: Clone> Tensor<T> {
    pub fn get_offset(&self) -> &usize {
        &self.offset
    }

    pub fn set_offset(&mut self, new: usize) -> Result<()> {
        if self.data.len() < new {
            return Err(TensorError::IndexOutOfBounds);
        }

        self.offset = new;
        Ok(())
    }

    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    pub fn set_data(&mut self, new: Vec<T>) -> Result<()> {
        if self.data.len() != new.len() {
            return Err(TensorError::DimensionalMismatch);
        }

        self.data = new;
        Ok(())
    }

    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn set_shape(&mut self, new: Vec<usize>) -> Result<()> {
        if self.shape.len() != new.len() {
            return Err(TensorError::ShapeMismatch);
        }

        self.shape = new;
        Ok(())
    }

    pub fn get_strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn set_strides(&mut self, new: Vec<usize>) -> Result<()> {
        if self.strides.len() != new.len() {
            return Err(TensorError::DimensionalMismatch);
        }

        self.strides = new;
        Ok(())
    }

    pub fn get_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds);
        }

        Ok(indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum())
    }

    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        if let Ok(index) = self.get_index(indices) {
            if let Some(val) = self.data.get(index) {
                return Ok(val);
            }
        }
        Err(TensorError::IndexOutOfBounds)
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T> {
        if let Ok(index) = self.get_index(indices) {
            if let Some(val) = self.data.get_mut(index) {
                return Ok(val);
            }
        }
        Err(TensorError::IndexOutOfBounds)
    }

    pub fn set(&mut self, indices: &[usize], new: T) -> Result<()> {
        if let Ok(value) = self.get_mut(indices) {
            *value = new;
            return Ok(());
        }
        Err(TensorError::IndexOutOfBounds)
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<Self> {
        if self.data.len() != new_shape.iter().product::<usize>() {
            println!("Total size must remain the same");
            return Err(TensorError::ShapeMismatch);
        }

        self.strides = compute_strides(&new_shape);
        Ok(self.clone())
    }

    pub fn new(shape: Vec<usize>, fill: T) -> Self {
        let total = shape.iter().product();
        let data = vec![fill; total];
        let strides = compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn from_data(shape: Vec<usize>, data: Vec<T>) -> Result<Self> {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape does not match data length"
        );
        if shape.iter().product::<usize>() != data.len() {
            return Err(TensorError::ShapeMismatch);
        }

        let strides = compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn from<const M: usize, const N: usize>(arr: [[T; N]; M]) -> Self {
        let mut data = Vec::with_capacity(M * N);
        for row in arr.iter() {
            data.extend_from_slice(row);
        }
        let shape = vec![M, N];
        let strides = compute_strides(&shape);
        let offset = 0;
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl<T: Copy> Tensor<T> {
    pub fn assert_same_shape(&self, other: &Self) {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
    }
}

impl<T: Clone> Tensor<T> {
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.get_shape().len() {
            return Err(TensorError::ShapeMismatch);
        }

        if !is_valid_perm(dims) {
            return Err(TensorError::InvalidPermutation);
        }

        let new_shape = dims.iter().map(|&i| self.get_shape()[i]).collect();
        let new_strides = dims.iter().map(|&i| self.get_strides()[i]).collect();

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: *self.get_offset(),
        })
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let mut dims: Vec<usize> = (0..self.shape.len()).collect();
        dims.swap(dim1, dim2);
        self.permute(&dims)
    }
}

fn is_valid_perm(dims: &[usize]) -> bool {
    let mut seen = vec![false; dims.len()];

    for &d in dims {
        if d >= dims.len() || seen[d] {
            return false;
        }
        seen[d] = true;
    }
    true
}

use std::ops::Sub;

impl<T: Sub<Output = T> + Clone + Copy> Sub for Tensor<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.elementwise_sub(&rhs)
    }
}
impl<T> Tensor<T>
where
    T: Copy + Sub<Output = T>,
{
    pub fn elementwise_sub(&self, rhs: &Self) -> Self {
        self.assert_same_shape(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

use std::ops::Mul;

impl<T> Tensor<T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(TensorError::ShapeMismatch);
        }

        let m = self.shape[0];
        let k1 = self.shape[1];
        let k2 = rhs.shape[0];
        let n = rhs.shape[1];

        if k1 != k2 {
            return Err(TensorError::DimensionalMismatch);
        }

        let mut result = Tensor::new(vec![m, n], T::default());

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();

                for k in 0..k1 {
                    let a = *self.get(&[i, k])?;
                    let b = *rhs.get(&[k, j])?;
                    sum = sum + (a * b);
                }
                //let mut x = result.get_mut(&[i, j])?;
                //x = &mut sum;
                let _ = result.set(&[i, j], sum);
            }
        }

        Ok(result)
    }
}
impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T>,
{
    pub fn elementwise_mul(&self, rhs: &Self) -> Self {
        self.assert_same_shape(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl<T: Mul<Output = T> + Clone + Copy> Mul for Tensor<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.elementwise_mul(&rhs)
    }
}

use std::ops::Div;

impl<T: Div<Output = T> + Clone + Copy> Div for Tensor<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.elementwise_div(&rhs)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Div<Output = T>,
{
    pub fn elementwise_div(&self, rhs: &Self) -> Self {
        self.assert_same_shape(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a / *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

use std::ops::Add;

impl<T: Add<Output = T> + Clone + Copy> Add for Tensor<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.elementwise_add(&rhs)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Add<Output = T>,
{
    pub fn elementwise_add(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch for addition");
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl Tensor<f32> {
    pub fn relu(&self) -> Result<Self> {
        let data = self.get_data().iter().map(|x| x.max(0.0)).collect();
        Self::from_data(self.get_shape().to_vec(), data)
    }
    pub fn sigmoid(&self) -> Result<Self> {
        let data = self
            .get_data()
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Self::from_data(self.get_shape().to_vec(), data)
    }
    pub fn tanh(&self) -> Result<Self> {
        let data = self.get_data().iter().map(|x| x.tanh()).collect();
        Self::from_data(self.get_shape().to_vec(), data)
    }

    // pub fn softmax(&self, axis: usize) -> Self {
    //     assert!(axis < self.shape.len(), "Invalid axis");
    //     unimplemented!("softmax activation")
    // }
}

impl Tensor<f32> {
    pub fn exp(&self) -> Result<Self> {
        let data = self.get_data().iter().map(|x| x.exp()).collect();
        Self::from_data(self.get_shape().to_vec(), data)
    }

    pub fn log(&self) -> Result<Self> {
        let data = self.get_data().iter().map(|x| x.ln()).collect();
        Self::from_data(self.get_shape().to_vec(), data)
    }

    pub fn powf(&self, exponent: f32) -> Result<Self> {
        let data = self.get_data().iter().map(|x| x.powf(exponent)).collect();
        Self::from_data(self.get_shape().to_vec(), data)
    }

    pub fn sqrt(&self) -> Result<Self> {
        let data = self.get_data().iter().map(|x| x.sqrt()).collect();
        Self::from_data(self.get_shape().to_vec(), data)
    }
}

impl<T: Clone + Copy + Default + Add<Output = T>> Tensor<T> {
    pub fn sum(&self, axis: Option<usize>) -> Result<Self> {
        match axis {
            Some(axis) => {
                if axis >= self.shape.len() {
                    return Err(TensorError::InvalidAxis);
                }

                let mut new_shape = self.shape.clone();
                new_shape[axis] = 1;

                let result = Tensor::new(new_shape, T::default());

                // Implement axis-wise summation
                // ...

                Ok(result)
            }
            None => {
                let sum = self.data.iter().fold(T::default(), |acc, &x| acc + x);
                Tensor::from_data(vec![1], vec![sum])
            }
        }
    }

    // pub fn mean(&self, axis: Option<usize>) -> Result<Self> {
    //     // Similar to sum but divide by number of elements
    //     unimplemented!()
    // }

    // pub fn max(&self, axis: Option<usize>) -> Result<Self> {
    //     // Find maximum values
    //     unimplemented!()
    // }

    // pub fn min(&self, axis: Option<usize>) -> Result<Self> {
    //     // Find minimum values
    //     unimplemented!()
    // }
}

impl<T> Tensor<T> {
    pub fn contiguous(&self) -> Self {
        // Ensure tensor is contiguous in memory
        // May require reordering elements
        unimplemented!()
    }

    pub fn is_contiguous(&self) -> bool {
        // Check if tensor is contiguous
        unimplemented!()
    }
}

// impl<T: Serialize> Tensor<T> {
//     pub fn to_json(&self) -> Result<String> {
//         serde_json::to_string(self).map_err(|_| TensorError::SerializationError)
//     }
// }

// impl<T: DeserializeOwned> Tensor<T> {
//     pub fn from_json(json: &str) -> Result<Self> {
//         serde_json::from_str(json).map_err(|_| TensorError::DeserializationError)
//     }
// }
