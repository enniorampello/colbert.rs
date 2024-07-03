use rand::Rng;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug)]
struct Matrix {
    dims: (usize, usize),
    data: Vec<f32>, // turn this into a Box<>?
}

#[derive(Debug)]
#[allow(dead_code)]
struct SizeError {
    message: String,
}

enum MatrixOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut output = String::new();

        output.push_str(format!("Matrix size: ({}, {})\n", self.dims.0, self.dims.1).as_str());
        for i in 0..self.dims.0 {
            output.push_str("[ ");
            for j in 0..self.dims.1 {
                output.push_str(format!("{: ^8.4} ", self.get(i, j)).as_str());
            }
            output.push_str("]\n");
        }
        write!(f, "{}", output)
    }
}

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, rhs: &'b Matrix) -> Matrix {
        self.broadcast_op(rhs, MatrixOp::Add)
    }
}

impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &'b Matrix) -> Matrix {
        self.broadcast_op(rhs, MatrixOp::Sub)
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Matrix {
        self.broadcast_op(rhs, MatrixOp::Mul)
    }
}

impl<'a, 'b> Div<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn div(self, rhs: &'b Matrix) -> Matrix {
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

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            dims: (rows, cols),
            data: vec![1f32; rows * cols],
        }
    }

    pub fn init_uniform(&mut self, low: f32, high: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..(self.dims.0 * self.dims.1) {
            self.data[i] = rng.gen_range(low..high);
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.dims.1 + col]
    }

    pub fn set(&mut self, row: usize, col: usize, res: f32) {
        self.data[row * self.dims.1 + col] = res;
    }

    pub fn broadcast_op(&self, other: &Matrix, op: MatrixOp) -> Matrix {
        if self.dims.0 != other.dims.0 {
            panic!(
                "Size mismatch during addition. {} != {}.",
                self.dims.0, other.dims.0
            );
        } else if self.dims.1 != other.dims.1 {
            panic!(
                "Size mismatch during addition. {} != {}.",
                self.dims.1, other.dims.1
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

        Matrix {
            dims: (self.dims.0, self.dims.1),
            data: result,
        }
    }

    pub fn apply_fn<F>(&self, f: F) -> Matrix
    where
        F: Fn(f32) -> f32,
    {
        Matrix {
            dims: (self.dims.0, self.dims.1),
            data: self.data.iter().cloned().map(f).collect(),
        }
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        if self.dims.1 != other.dims.0 {
            panic!(
                "Size mismatch: {} different from {}.",
                self.dims.1, other.dims.0
            );
        }

        let mut result = Matrix::new(self.dims.0, other.dims.1);

        for i in 0..self.dims.0 {
            for j in 0..other.dims.1 {
                result.data[i * other.dims.1 + j] = dot(
                    self.data[i * self.dims.1..(i + 1) * self.dims.1].to_vec(),
                    other
                        .data
                        .iter()
                        .skip(j)
                        .step_by(other.dims.1)
                        .copied()
                        .collect(),
                )
            }
        }

        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.dims.1, self.dims.0);

        for i in 0..self.dims.0 {
            for j in 0..self.dims.1 {
                result.set(j, i, self.get(i, j));
            }
        }

        result
    }
}

fn main() {
    let mut a = Matrix::new(3, 2);
    let b = Matrix::new(3, 2);
    let sq = |n: f32| n.powi(2);

    a.init_uniform(-1.0, 1.0);

    let c = &a - &b;
    let d = a.mul(&b.transpose());
    let e = c.apply_fn(sq);
    // let d = a.mul(&c);

    print!("{}", a);
    print!("{}", b);
    print!("{}", c);
    print!("{}", d);
    print!("{}", e);
}
