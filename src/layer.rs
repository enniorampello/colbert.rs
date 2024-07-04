use rand::Rng;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug)]
pub struct Layer {
    shape: (usize, usize), // (input_size, output_size)
    data: Vec<f32>,        // turn this into a Box<>?
}

pub enum MatrixOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut output = String::new();

        output.push_str(format!("Matrix size: ({}, {})\n", self.shape.0, self.shape.1).as_str());
        for i in 0..self.shape.0 {
            output.push_str("[ ");
            for j in 0..self.shape.1 {
                output.push_str(format!("{: ^8.4} ", self.get(i, j)).as_str());
            }
            output.push_str("]\n");
        }
        write!(f, "{}", output)
    }
}

impl<'a, 'b> Add<&'b Layer> for &'a Layer {
    type Output = Layer;

    fn add(self, rhs: &'b Layer) -> Layer {
        self.broadcast_op(rhs, MatrixOp::Add)
    }
}

impl<'a, 'b> Sub<&'b Layer> for &'a Layer {
    type Output = Layer;

    fn sub(self, rhs: &'b Layer) -> Layer {
        self.broadcast_op(rhs, MatrixOp::Sub)
    }
}

impl<'a, 'b> Mul<&'b Layer> for &'a Layer {
    type Output = Layer;

    fn mul(self, rhs: &'b Layer) -> Layer {
        self.broadcast_op(rhs, MatrixOp::Mul)
    }
}

impl<'a, 'b> Div<&'b Layer> for &'a Layer {
    type Output = Layer;

    fn div(self, rhs: &'b Layer) -> Layer {
        self.broadcast_op(rhs, MatrixOp::Div)
    }
}

pub fn dot(v1: Vec<f32>, v2: Vec<f32>) -> f32 {
    let mut result = 0f32;

    if v1.len() != v2.len() {
        panic!(
            "Size mismatch in vector mul: len(v1)={}, len(v2)={}",
            v1.len(),
            v2.len()
        )
    }

    for n in 0..v1.len() {
        result += v1[n] * v2[n];
    }

    result
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        Self {
            shape: (nin, nout),
            data: vec![1f32; nin * nout],
        }
    }

    pub fn init_uniform(&mut self, low: f32, high: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..(self.shape.0 * self.shape.1) {
            self.data[i] = rng.gen_range(low..high);
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.shape.1 + col]
    }

    pub fn set(&mut self, row: usize, col: usize, res: f32) {
        self.data[row * self.shape.1 + col] = res;
    }

    pub fn broadcast_op(&self, other: &Layer, op: MatrixOp) -> Layer {
        if self.shape.0 != other.shape.0 {
            panic!(
                "Size mismatch during addition. {} != {}.",
                self.shape.0, other.shape.0
            );
        } else if self.shape.1 != other.shape.1 {
            panic!(
                "Size mismatch during addition. {} != {}.",
                self.shape.1, other.shape.1
            );
        }
        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| match op {
                MatrixOp::Add => a + b,
                MatrixOp::Sub => a - b,
                MatrixOp::Mul => a * b,
                MatrixOp::Div => a / b,
            })
            .collect();

        Layer {
            shape: (self.shape.0, self.shape.1),
            data: result,
        }
    }

    pub fn apply_fn<F>(&self, f: F) -> Layer
    where
        F: Fn(f32) -> f32,
    {
        Layer {
            shape: (self.shape.0, self.shape.1),
            data: self.data.iter().cloned().map(f).collect(), // this might not be the optimal way to do this. should I test multithreading?
        }
    }

    pub fn mul(&self, other: &Layer) -> Layer {
        if self.shape.1 != other.shape.0 {
            panic!(
                "Size mismatch: {} different from {}.",
                self.shape.1, other.shape.0
            );
        }

        let mut result = Layer::new(self.shape.0, other.shape.1);

        for i in 0..self.shape.0 {
            for j in 0..other.shape.1 {
                result.data[i * other.shape.1 + j] = dot(
                    self.data[i * self.shape.1..(i + 1) * self.shape.1].to_vec(),
                    other
                        .data
                        .iter()
                        .skip(j)
                        .step_by(other.shape.1)
                        .copied()
                        .collect(),
                )
            }
        }

        result
    }

    pub fn transpose(&self) -> Layer {
        let mut result = Layer::new(self.shape.1, self.shape.0);

        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                result.set(j, i, self.get(i, j));
            }
        }

        result
    }
}
