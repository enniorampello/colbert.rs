use std::collections::LinkedList;

use crate::layer::Layer;

pub struct Sequential {
    layers: LinkedList<Layer>,
}

pub trait Backprop {
    fn backward(&mut self);
}
