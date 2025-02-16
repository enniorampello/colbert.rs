use crate::matrix::Matrix;

pub mod matrix;

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
