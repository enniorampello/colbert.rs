use rand::Rng;

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

impl std::fmt::Display for Matrix {
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

pub fn vec_mul(v1: Vec<f32>, v2: Vec<f32>) -> f32 {
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
        Matrix {
            dims: (rows, cols),
            data: vec![0f32; rows * cols],
        }
    }

    pub fn init_uniform(&mut self, low: f32, high: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..(self.dims.0 * self.dims.1) {
            self.data[i] = rng.gen_range(low..high);
        }
        self.transpose();
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.dims.1 + col]
    }

    pub fn set(&mut self, row: usize, col: usize, res: f32) {
        self.data[row * self.dims.1 + col] = res;
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
                result.data[i * other.dims.1 + j] = vec_mul(
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
    a.init_uniform(-1.0, 1.0);

    let c = a.transpose();

    let d = a.mul(&c);

    print!("{}", a);
    print!("{}", c);
    print!("{}", d);
}
