# colbert.rs
Rust implementation of the ColBERT architecture.

## Quickstart
At the moment this repo only contains a matrix library written in pure Rust (without using any external crates).

## ToDos
- [x] implement matrix data structure and basic matrix operations.
- [ ] implement MLP layer with forward and back prop
- [ ] offload matrix operations to the GPU using `wgpu` crate
- [ ] implement inference for ColBERT v1
- [ ] implement inference for ColBERT v2
- [ ] implement training for ColBERT v2
